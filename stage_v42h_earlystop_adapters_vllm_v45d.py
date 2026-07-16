#!/usr/bin/env python3
"""Audit and stage sealed V42H step-16/32 adapters for compact V45D."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v45a as prior


ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = (
    ROOT / "experiments/sft_controls/"
    "v42h_matched_init_equal_unit_fold3_v412_lr6e5/"
    "middle_late_r32_seed17_init20260715041"
).resolve()
REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42h_matched_init_equal_unit_fold3_v412_lr6e5/runtime_report_v42h.json"
).resolve()
COMMON = {
    "report": REPORT,
    "seal": "45fea5d0bae1658b4ef388a501238471ab894876b108cca9cd514143140fcc2b",
    "seal_content": "275a645c4b7a8689649390c408c0767c3cd09c679b609e4c32d0e6dcf573088f",
    "schema": "specialist-sft-matched-init-equal-unit-runtime-v42h",
    "status": "complete_train_only_lr6e5_state_sealed_shadow_unopened",
    "learning_rate": 6e-5,
    "learning_rate_label": "6e-5",
}
SOURCE_SPECS_V45D = {
    "sft_v42h_step16": {
        **COMMON,
        "source": TRAIN_ROOT / "checkpoint-16",
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42h_step16_sft_qwen35_vllm_namespace_v45d"
        ).resolve(),
        "artifact_prefix": "checkpoint-16",
        "weights": "9a9f5ef88a4afe4dd1ee9cb564f74977fa57daa73c5d8e88b08e56739da6d4da",
        "config": "8da87ae343a460345b074da23be3dceddb263b459e64cd5db1d14b02a4905ef7",
        "completed_steps": 16,
        "expected_epoch": 1.0,
        "trainer_state_sha256": "c5424cbec1a91bff26a787c6110c700f3a25b395236f386f0815b00eead4c05d",
    },
    "sft_v42h_step32": {
        **COMMON,
        "source": TRAIN_ROOT / "checkpoint-32",
        "output": (
            ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
            "v42h_step32_sft_qwen35_vllm_namespace_v45d"
        ).resolve(),
        "artifact_prefix": "checkpoint-32",
        "weights": "940c9c0da1539537c02a402c0768fb1c8e65321bef4683df2dd42a7b0a12c2ea",
        "config": "8da87ae343a460345b074da23be3dceddb263b459e64cd5db1d14b02a4905ef7",
        "completed_steps": 32,
        "expected_epoch": 2.0,
        "trainer_state_sha256": "edb5c4192dd148a33aad6564c4db7e3ad6e7761a42b87d969999f7f2b0d9402e",
    },
}


@contextmanager
def injected_spec_v45d(arm: str):
    if arm not in SOURCE_SPECS_V45D:
        raise ValueError(f"unknown V45D early-stop arm: {arm}")
    previous = prior.SOURCE_SPECS_V45A.get(arm)
    prior.SOURCE_SPECS_V45A[arm] = SOURCE_SPECS_V45D[arm]
    try:
        yield
    finally:
        if previous is None:
            prior.SOURCE_SPECS_V45A.pop(arm, None)
        else:
            prior.SOURCE_SPECS_V45A[arm] = previous


def audit_source_v45d(arm: str) -> dict:
    spec = SOURCE_SPECS_V45D[arm]
    with injected_spec_v45d(arm):
        result = prior.audit_source_v45a(arm)
    state_path = spec["source"] / "trainer_state.json"
    state = json.loads(state_path.read_text())
    report = json.loads(spec["report"].read_text())
    artifact_hash = report["artifacts"]["output_file_sha256"].get(
        f'{spec["artifact_prefix"]}/trainer_state.json'
    )
    if (
        prior.prior.file_sha256_v44a(state_path)
        != spec["trainer_state_sha256"]
        or artifact_hash != spec["trainer_state_sha256"]
        or state.get("global_step") != spec["completed_steps"]
        or state.get("epoch") != spec["expected_epoch"]
        or state.get("max_steps") != 48
        or state.get("num_train_epochs") != 3
        or state.get("best_model_checkpoint") is not None
        or state.get("best_metric") is not None
    ):
        raise RuntimeError(f"V45D {arm} trainer-state step/epoch changed")
    return {
        **result,
        "trainer_state_binding_v45d": {
            "file_sha256": spec["trainer_state_sha256"],
            "global_step": state["global_step"],
            "epoch": state["epoch"],
            "max_steps": state["max_steps"],
            "num_train_epochs": state["num_train_epochs"],
        },
    }


def stage_one_v45d(arm: str, output: Path | None = None) -> dict:
    audit_source_v45d(arm)
    with injected_spec_v45d(arm):
        return prior.stage_one_v45a(arm, output)


def main() -> int:
    if any(spec["output"].exists() for spec in SOURCE_SPECS_V45D.values()):
        raise FileExistsError("V45D staged adapter output already exists")
    created: list[Path] = []
    try:
        result = {}
        for arm, spec in SOURCE_SPECS_V45D.items():
            manifest = stage_one_v45d(arm)
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
