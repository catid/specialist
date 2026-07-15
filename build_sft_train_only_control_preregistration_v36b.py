#!/usr/bin/env python3
"""Build V36B: a fresh attempt after V36A's pre-step API failure."""

from __future__ import annotations

import json
from pathlib import Path

import build_sft_train_only_control_preregistration_v36a as base
import run_sft_train_only_control_v36a as runtime


ROOT = Path(__file__).resolve().parent
RUN_DIR = (ROOT / "experiments/sft_controls/v36b_v412").resolve()
DATASET = (
    ROOT / "experiments/sft_controls/v36a_v412/train_v412.jsonl"
).resolve()
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17_attempt_b"
PREREGISTRATION = RUN_DIR / "preregistration_v36b.json"
FAILED_ATTEMPT = (
    ROOT / "experiments/sft_controls/v36a_v412/attempt_v36a.json"
).resolve()
EXPECTED = {
    **base.EXPECTED,
    "runner_sha256": "e9823359d82aeb61d7fd1f17f24ea22fc46680cc122c55232b1913d765d1d49d",
    "sft_sha256": "cbb3a2767c0ec098528cd7f71dbef87d1d9f737c2d4fc5486ba590ae2a19f636",
}
FAILED_ATTEMPT_FILE_SHA256 = (
    "a3f8cb4fed7b55fce3c7aaeb324ea8c1ba2cda1ed07f734232cd851cdfddc570"
)
FAILED_ATTEMPT_CONTENT_SHA256 = (
    "3cfba75e64dcb599acdabc67db4c893c67e81d46cc708b4a3162c94b4f1bbc0a"
)


def _configure_base():
    base.RUN_DIR = RUN_DIR
    base.DATASET = DATASET
    base.OUTPUT_DIR = OUTPUT_DIR
    base.PREREGISTRATION = PREREGISTRATION
    base.ARTIFACT_TAG = "v36b"
    base.EXPECTED = EXPECTED


def _arguments():
    _configure_base()
    return base._arguments()


def build():
    _configure_base()
    if runtime.file_sha256(FAILED_ATTEMPT) != FAILED_ATTEMPT_FILE_SHA256:
        raise RuntimeError("v36b failed-attempt file identity changed")
    failed = json.loads(FAILED_ATTEMPT.read_text(encoding="utf-8"))
    if (
        failed.get("content_sha256_before_self_field")
        != FAILED_ATTEMPT_CONTENT_SHA256
        or failed.get("status") != "failed"
        or failed.get("phase") != "child_complete"
        or failed.get("returncode") != 1
    ):
        raise RuntimeError("v36b failed-attempt content changed")
    result = base.build()
    result["experiment_name"] = (
        "sft_train_only_v36b_v412_middle_late_r32_seed17"
    )
    result["retry_lineage"] = {
        "prior_attempt": str(FAILED_ATTEMPT),
        "prior_attempt_file_sha256": FAILED_ATTEMPT_FILE_SHA256,
        "prior_attempt_content_sha256": FAILED_ATTEMPT_CONTENT_SHA256,
        "prior_optimizer_steps": 0,
        "prior_model_candidate_written": False,
        "failure": (
            "Transformers 5.12 rejected the obsolete group_by_length "
            "TrainingArguments keyword before Trainer construction"
        ),
        "isolated_change": (
            "remove group_by_length and assert all TrainingArguments keys "
            "are accepted by the live installed signature"
        ),
    }
    result["content_sha256_before_self_field"] = runtime.canonical_sha256({
        key: item for key, item in result.items()
        if key != "content_sha256_before_self_field"
    })
    return result


def main():
    result = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PREREGISTRATION.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(PREREGISTRATION)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
