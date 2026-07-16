#!/usr/bin/env python3
"""Audit and stage sealed V42E/F adapters in vLLM's language_model namespace."""

from __future__ import annotations

import json
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
SOURCE_SPECS_V45A = {
    "sft_v42b_step16": {
        "source": (
            ROOT / "experiments/sft_controls/"
            "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
            "middle_late_r32_seed17_init20260715041/checkpoint-16"
        ).resolve(),
        "report": (
            ROOT / "experiments/sft_controls/"
            "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
            "runtime_report_v42b.json"
        ).resolve(),
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42b_step16_sft_qwen35_vllm_namespace_v45a"
        ).resolve(),
        "artifact_prefix": "checkpoint-16",
        "weights": "1d4b6c795bcdf57964c08853ca90243dad5ad852347dde98b51c021fc88a1656",
        "config": "60bd216e895cba461a7cda482dc72635ac4037f301ee0f0c295e00d5a1247f5d",
        "seal": "303a9ecfdcb32ad49f16a990e47ee1ab85f5e0d026427c91ea3fadc16567e7e7",
        "seal_content": "beeb062797d7ea387857566bb955d638ab672920c64af3a44d7a49b607dbef13",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42b",
        "status": "complete_train_only_retry_state_sealed_shadow_unopened",
        "learning_rate": 1e-4,
        "learning_rate_label": "1e-4",
        "completed_steps": 16,
        "trainer_state_sha256": "0b161cf9953bc40e6c8d20d60297202bf44a53364f87a16dc402dd54bbaa425f",
    },
    "sft_v42b_step32": {
        "source": (
            ROOT / "experiments/sft_controls/"
            "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
            "middle_late_r32_seed17_init20260715041/checkpoint-32"
        ).resolve(),
        "report": (
            ROOT / "experiments/sft_controls/"
            "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
            "runtime_report_v42b.json"
        ).resolve(),
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42b_step32_sft_qwen35_vllm_namespace_v45a"
        ).resolve(),
        "artifact_prefix": "checkpoint-32",
        "weights": "670a830955192e3a090c57df3a720cbec98079ee74b975f8983ecc3c75c479c4",
        "config": "60bd216e895cba461a7cda482dc72635ac4037f301ee0f0c295e00d5a1247f5d",
        "seal": "303a9ecfdcb32ad49f16a990e47ee1ab85f5e0d026427c91ea3fadc16567e7e7",
        "seal_content": "beeb062797d7ea387857566bb955d638ab672920c64af3a44d7a49b607dbef13",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42b",
        "status": "complete_train_only_retry_state_sealed_shadow_unopened",
        "learning_rate": 1e-4,
        "learning_rate_label": "1e-4",
        "completed_steps": 32,
        "trainer_state_sha256": "73f89186be2038224225b635b0315e24336e05011f91e4162d14494c925b08c2",
    },
    "sft_v42e": {
        "source": (
            ROOT / "experiments/sft_controls/"
            "v42e_matched_init_equal_unit_fold3_v412_lr3e4/"
            "middle_late_r32_seed17_init20260715041/final"
        ).resolve(),
        "report": (
            ROOT / "experiments/sft_controls/"
            "v42e_matched_init_equal_unit_fold3_v412_lr3e4/runtime_report_v42e.json"
        ).resolve(),
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42e_sft_qwen35_vllm_namespace_v45a"
        ).resolve(),
        "weights": "98f4dbed55eb1b016cbf0d0a785450ac3a9a35173916c312f024af21b8ce5df9",
        "config": "3b536e46822842c3627e02077dcb3d6e76d1c22f3bcc5746a70744356ef3c2a2",
        "seal": "ae741a2f526fa44515df67f9aa4caaf04ed222b87fd7a9529d14c5e0dba36c9a",
        "seal_content": "02fd8cd8e6c3636b844cb888696c662cb5a68e344ef8b0b488ea62edba63c5c2",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42e",
        "status": "complete_train_only_lr3e4_state_sealed_shadow_unopened",
        "learning_rate": 3e-4,
        "learning_rate_label": "3e-4",
        "artifact_prefix": "final",
        "completed_steps": 48,
    },
    "sft_v42f": {
        "source": (
            ROOT / "experiments/sft_controls/"
            "v42f_matched_init_equal_unit_fold3_v412_lr1e3/"
            "middle_late_r32_seed17_init20260715041/final"
        ).resolve(),
        "report": (
            ROOT / "experiments/sft_controls/"
            "v42f_matched_init_equal_unit_fold3_v412_lr1e3/runtime_report_v42f.json"
        ).resolve(),
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42f_sft_qwen35_vllm_namespace_v45a"
        ).resolve(),
        "weights": "6ad5a43627ef20d85f960dc74749ac739cf0edd34c0489e23cd2cecdaed660c9",
        "config": "13ddbd761d4000a453ee8d5361a1cce9a4c3595c44a85f8d485230cffbd67ae9",
        "seal": "c47bc8ebeabb71843edbd646409de04dfa5e43f7043962e2fb682dd7bc747b28",
        "seal_content": "e440a307d9bd9bd2984432798bb31a831782966f1296a9d301c80cff11e83ef3",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42f",
        "status": "complete_train_only_lr1e3_state_sealed_shadow_unopened",
        "learning_rate": 1e-3,
        "learning_rate_label": "1e-3",
        "artifact_prefix": "final",
        "completed_steps": 48,
    },
    "sft_v42g": {
        "source": (
            ROOT / "experiments/sft_controls/"
            "v42g_matched_init_equal_unit_fold3_v412_lr5e5/"
            "middle_late_r32_seed17_init20260715041/final"
        ).resolve(),
        "report": (
            ROOT / "experiments/sft_controls/"
            "v42g_matched_init_equal_unit_fold3_v412_lr5e5/runtime_report_v42g.json"
        ).resolve(),
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42g_sft_qwen35_vllm_namespace_v45a"
        ).resolve(),
        "artifact_prefix": "final",
        "weights": "b5820bd3a01605bbd7c488900bdeb27dc2dbaf5073e9712edfbff0f7402f3bd0",
        "config": "99548642c32678b1a69f0e2c1151c96f10b87b821e6588feefebd353684f2178",
        "seal": "82633c37a17e597961ca8a69d2a1a07218dbd6d0df53d9ebb75217ba54549430",
        "seal_content": "5c9f0d80061b0cd6f95ac401802fe96fdff42e585c94737a536a65186bd66986",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42g",
        "status": "complete_train_only_lr5e5_state_sealed_shadow_unopened",
        "learning_rate": 5e-5,
        "learning_rate_label": "5e-5",
        "completed_steps": 48,
    },
}

STAGED_BY_ARM_V45A = {
    arm: spec[2] for arm, spec in prior.CANDIDATE_SPECS_V44A.items()
}
STAGED_BY_ARM_V45A.update({
    arm: spec["output"] for arm, spec in SOURCE_SPECS_V45A.items()
})


def _source_seal_v45a(arm: str) -> dict:
    spec = SOURCE_SPECS_V45A[arm]
    if prior.file_sha256_v44a(spec["report"]) != spec["seal"]:
        raise RuntimeError(f"V45A {arm} report file changed")
    report = json.loads(spec["report"].read_text())
    compact = {
        key: item for key, item in report.items()
        if key != "content_sha256_before_self_field"
    }
    artifacts = report.get("artifacts", {}).get("output_file_sha256", {})
    prefix = spec["artifact_prefix"]
    if (
        report.get("content_sha256_before_self_field") != spec["seal_content"]
        or prior.canonical_sha256_v44a(compact) != spec["seal_content"]
        or report.get("schema") != spec["schema"]
        or report.get("status") != spec["status"]
        or report.get("validation_ood_or_holdout_opened") is not False
        or artifacts.get(f"{prefix}/adapter_model.safetensors")
        != spec["weights"]
        or artifacts.get(f"{prefix}/adapter_config.json") != spec["config"]
    ):
        raise RuntimeError(f"V45A {arm} train-only source seal changed")
    return {
        "schema": "matched-sft-checkpoint-success-seal-v45a",
        "report": str(spec["report"]),
        "report_file_sha256": spec["seal"],
        "report_content_sha256": spec["seal_content"],
        "artifact_prefix": prefix,
        "completed_steps": spec["completed_steps"],
        "state_complete": True,
        "selection_data_opened": False,
    }


def _injected_source_v45a(arm: str):
    if arm not in SOURCE_SPECS_V45A:
        raise ValueError(f"unknown V45A source arm: {arm}")
    spec = SOURCE_SPECS_V45A[arm]
    old_candidate = prior.CANDIDATE_SPECS_V44A.get(arm)
    old_expected = prior.EXPECTED_V44A.get(arm)
    old_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[arm] = (
        spec["source"], spec["report"], spec["output"]
    )
    prior.EXPECTED_V44A[arm] = {
        key: spec[key] for key in (
            "weights", "config", "seal", "seal_content", "schema", "status"
        )
    }
    prior._source_seal_v44a = lambda requested: (
        _source_seal_v45a(requested)
        if requested == arm else old_seal(requested)
    )
    return old_candidate, old_expected, old_seal


def _restore_source_v45a(arm: str, saved) -> None:
    old_candidate, old_expected, old_seal = saved
    if old_candidate is None:
        prior.CANDIDATE_SPECS_V44A.pop(arm, None)
    else:
        prior.CANDIDATE_SPECS_V44A[arm] = old_candidate
    if old_expected is None:
        prior.EXPECTED_V44A.pop(arm, None)
    else:
        prior.EXPECTED_V44A[arm] = old_expected
    prior._source_seal_v44a = old_seal


def audit_source_v45a(arm: str) -> dict:
    saved = _injected_source_v45a(arm)
    try:
        result = prior.audit_source_v44a(arm)
        report = json.loads(SOURCE_SPECS_V45A[arm]["report"].read_text())
        recipe = report.get("recipe", {})
        expected_lr = SOURCE_SPECS_V45A[arm]["learning_rate"]
        initialization = recipe.get("initialization", {})
        spec = SOURCE_SPECS_V45A[arm]
        if (
            recipe.get("learning_rate") != expected_lr
            or (
                arm not in ("sft_v42b_step16", "sft_v42b_step32")
                and recipe.get("only_change_from_v42b")
                != (
                    "peak cosine-schedule learning rate 1e-4 -> "
                    + SOURCE_SPECS_V45A[arm]["learning_rate_label"]
                )
            )
            or recipe.get("loss_mode")
            != "equal_conflict_unit_answer_token_mean"
            or recipe.get("prompt_mode") != "es_exact"
            or recipe.get("world_size") != 4
            or recipe.get("target_layers") != "20,21,22,23"
            or initialization.get("sealed_initialization_seed") != 20260715041
            or initialization.get("tensor_count") != 70
            or initialization.get("elements") != 4_528_128
            or initialization.get("unscaled_fp32_master") is not True
        ):
            raise RuntimeError(f"V45A {arm} matched training evidence changed")
        if "trainer_state_sha256" in spec:
            state_path = spec["source"] / "trainer_state.json"
            state = json.loads(state_path.read_text())
            report_hash = report["artifacts"]["output_file_sha256"].get(
                f'{spec["artifact_prefix"]}/trainer_state.json'
            )
            if (
                prior.file_sha256_v44a(state_path)
                != spec["trainer_state_sha256"]
                or report_hash != spec["trainer_state_sha256"]
                or state.get("global_step") != spec["completed_steps"]
                or state.get("max_steps") != 48
                or state.get("epoch") != spec["completed_steps"] / 16
            ):
                raise RuntimeError(f"V45A {arm} early-stop state changed")
        return result
    finally:
        _restore_source_v45a(arm, saved)


def stage_one_v45a(arm: str, output: Path | None = None) -> dict:
    # Audit through this wrapper first so learning-rate/matched-init evidence is
    # checked in addition to the canonical 70-tensor byte inventory.
    audit_source_v45a(arm)
    saved = _injected_source_v45a(arm)
    try:
        return prior.stage_one_v44a(
            arm, Path(output).resolve() if output is not None else None
        )
    finally:
        _restore_source_v45a(arm, saved)


def main() -> int:
    if any(spec["output"].exists() for spec in SOURCE_SPECS_V45A.values()):
        raise FileExistsError("V45A staged adapter output already exists")
    created = []
    try:
        result = {}
        for arm, spec in SOURCE_SPECS_V45A.items():
            manifest = stage_one_v45a(arm)
            created.append(spec["output"])
            result[arm] = {
                "directory": manifest["artifact"]["directory"],
                "weights_sha256": manifest["artifact"]["weights_file_sha256"],
                "manifest_content_sha256": manifest[
                    "content_sha256_before_self_field"
                ],
                "transformed_identity_sha256": manifest[
                    "transformed_identity"
                ]["sha256"],
            }
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException:
        for directory in reversed(created):
            for name in (
                "stage_manifest_v44a.json", "adapter_config.json",
                "adapter_model.safetensors",
            ):
                (directory / name).unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
