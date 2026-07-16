#!/usr/bin/env python3
"""Write the deterministic, CPU-only V49A sampling/weighting design artifact."""

from __future__ import annotations

import json
from pathlib import Path

import run_sft_train_only_control_v36a as engine
import v49a_train_only_source_temperature_weighting as design


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "train_only_capped_source_temperature_weighting_v49a.json"
).resolve()


def build() -> dict:
    value = design.analyze()
    value.update({
        "artifact_role": "preregisterable_design_only_no_launch_authority",
        "gpu_launch_authorized": False,
        "training_launch_authorized": False,
        "evaluation_launch_authorized": False,
        "implementation_bindings": {
            "design_runtime": engine.file_sha256(Path(design.__file__).resolve()),
            "builder": engine.file_sha256(Path(__file__).resolve()),
            "tests": engine.file_sha256(
                ROOT / "test_train_only_source_temperature_weighting_v49a.py"
            ),
            "curated_merge_runtime": engine.file_sha256(
                Path(design.curated.__file__).resolve()
            ),
            "conflict_graph_runtime": engine.file_sha256(
                Path(design.frozen.__file__).resolve()
            ),
            "equal_unit_runtime": engine.file_sha256(
                Path(design.equal_unit.__file__).resolve()
            ),
        },
        "running_experiment_isolation": {
            "existing_ab_test_read_or_mutated": False,
            "model_or_adapter_artifacts_read_or_mutated": False,
            "gpu_process_started": False,
        },
    })
    value["content_sha256_before_self_field"] = engine.canonical_sha256(value)
    return value


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("V49A refuses to overwrite its design artifact")
    value = build()
    engine.atomic_write_json(OUTPUT, value)
    print(json.dumps({
        "path": str(OUTPUT),
        "file_sha256": engine.file_sha256(OUTPUT),
        "content_sha256": value["content_sha256_before_self_field"],
        "merits_later_hpo_arm": value["recommendation"][
            "merits_one_later_preregistered_hpo_arm"
        ],
        "gpu_launch_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
