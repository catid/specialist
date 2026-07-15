#!/usr/bin/env python3
"""Build the content-free production/v401 replacement-fraction frame V34B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

import build_eggroll_es_joint_panels_v33a as hardened_v33
import eggroll_es_train_panel_sampler_v13 as sampler_v13


ROOT = Path(__file__).resolve().parent
CANDIDATE_PATH = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v401/"
    "train_qa_context_merit_v401.jsonl"
).resolve()
CANDIDATE_MANIFEST_PATH = (CANDIDATE_PATH.parent / "manifest.json").resolve()
CANDIDATE_BUILDER_PATH = (ROOT / "build_eggroll_es_v401_train_only_candidate_v34a.py").resolve()
CANDIDATE_TEST_PATH = (ROOT / "test_build_eggroll_es_v401_train_only_candidate_v34a.py").resolve()
PRODUCTION_PATH = (ROOT / "data/train_qa_curated_v1.jsonl").resolve()
OUTPUT_PATH = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v34b/"
    "production_v401_replacement_fraction_panels_v34b.json"
).resolve()

CANDIDATE_FREEZE_COMMIT = "59dfe718a914be8b37e05ff9daa822ab467d18a4"
CANDIDATE_SHA256 = "8e29826dd389171c69f5eb6f43781f900345974c3d4d11274268e86c6145693b"
CANDIDATE_MANIFEST_SHA256 = "1013032a1a4c21a2ece6e80e0930aecee8430639e07eaf48c2ac7701708e8f52"
CANDIDATE_MANIFEST_CONTENT_SHA256 = "42304107e89119c10c545dc79b4f85ab08bd4b8b78efec710d286987a3e8a5af"
CANDIDATE_BUILDER_SHA256 = "d62c8e21492432c81c8e83e281cc6a9cc48ba9b0c421ce875788fa85e4716de4"
CANDIDATE_TEST_SHA256 = "56b2f4a8c59a6e4ba981e8f3a01beb4a765e2bc5f6d01568e639d4a07d13d689"
CANDIDATE_ROWS = 531
PRODUCTION_SHA256 = "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
PRODUCTION_SOURCE_COMMIT = "a21de35748054c3ae8737a767606234952f9561e"
PRODUCTION_ROWS = 784
HARDENED_V33_BUILDER_SHA256 = "129ede4440fd13325edee309bf70d7e0e2356e42dacbac9aed0f4243fe54cbaf"
SAMPLER_SHA256 = "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"

# These are deliberately inherited from the hardened V33 joint-panel mechanics.
PANEL_SEED = hardened_v33.PANEL_SEED
REPRESENTATIVE_SEED = hardened_v33.REPRESENTATIVE_SEED
PANEL_NAMES = hardened_v33.PANEL_NAMES
OPTIMIZATION_PANELS = hardened_v33.OPTIMIZATION_PANELS
TRAIN_SCREENS = hardened_v33.TRAIN_SCREENS
STRATUM_QUOTAS = hardened_v33.STRATUM_QUOTAS
PANEL_SIZE = hardened_v33.PANEL_SIZE
REQUIRED_PAIRED_STRATA = hardened_v33.REQUIRED_PAIRED_STRATA
EXPECTED_JOINT_COMPONENTS = 326
EXPECTED_PAIRED_UNITS = 205
EXPECTED_CANDIDATE_ONLY_UNITS = 54
EXPECTED_PRODUCTION_ONLY_UNITS = 67
EXPECTED_PAIRED_STRATA = {
    "safety_consent": 62,
    "technique": 54,
    "equipment_material": 13,
    "resources_general": 76,
}

EXPECTED_PAIRED_ASSIGNMENT_ROOT_SHA256 = (
    "8a787f9845e88c52da658a94b3bd59ea5dc239c5baf6571e8e6993ac859a3e1f"
)
EXPECTED_SELECTED_UNIT_ROOT_SHA256 = (
    "67f18e167b0df5ccea9aeed1fd71276a121703978664b7364ef718befe36af07"
)
EXPECTED_RUNTIME_FRAME_CONTENT_SHA256 = (
    "a4f290bcdece10de81997d680c07475266896c7140ed394ede990b0e93d98c0e"
)
EXPECTED_AGGREGATE_CONTENT_SHA256 = (
    "2b2a5f9b66b59401f7ca1fffd4d901933d02b84b580356dc50b7b874da18dc7b"
)


canonical_sha256 = hardened_v33.canonical_sha256
file_sha256 = hardened_v33.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _git_blob(path: Path, commit: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
    )


def _read_jsonl_train_only(path: Path) -> list[dict]:
    # Row content exists only in memory while constructing the content-free frame.
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_bound_rows() -> tuple[list[dict], list[dict]]:
    expected = {
        CANDIDATE_PATH: CANDIDATE_SHA256,
        CANDIDATE_MANIFEST_PATH: CANDIDATE_MANIFEST_SHA256,
        CANDIDATE_BUILDER_PATH: CANDIDATE_BUILDER_SHA256,
        CANDIDATE_TEST_PATH: CANDIDATE_TEST_SHA256,
        PRODUCTION_PATH: PRODUCTION_SHA256,
        Path(hardened_v33.__file__).resolve(): HARDENED_V33_BUILDER_SHA256,
        Path(sampler_v13.__file__).resolve(): SAMPLER_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v34b bound train-only source or joint mechanics changed")
    for path in (
        CANDIDATE_PATH,
        CANDIDATE_MANIFEST_PATH,
        CANDIDATE_BUILDER_PATH,
        CANDIDATE_TEST_PATH,
    ):
        if hashlib.sha256(_git_blob(path, CANDIDATE_FREEZE_COMMIT)).hexdigest() != expected[path]:
            raise RuntimeError("v34b committed v401 identity changed")
    if hashlib.sha256(_git_blob(PRODUCTION_PATH, PRODUCTION_SOURCE_COMMIT)).hexdigest() != PRODUCTION_SHA256:
        raise RuntimeError("v34b committed production identity changed")
    freeze = json.loads(CANDIDATE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "eggroll-es-v401-train-only-candidate-manifest-v34a"
        or freeze.get("content_sha256_before_self_field")
        != CANDIDATE_MANIFEST_CONTENT_SHA256
        or canonical_sha256(_without_self(freeze))
        != CANDIDATE_MANIFEST_CONTENT_SHA256
        or freeze.get("candidate") != {
            "relative_path": str(CANDIDATE_PATH.relative_to(ROOT)),
            "file_sha256": CANDIDATE_SHA256,
            "rows": CANDIDATE_ROWS,
        }
        or freeze.get("train_only_replay", {}).get(
            "validation_heldout_ood_or_benchmark_file_opened"
        ) is not False
        or freeze.get("contains_row_content") is not False
    ):
        raise RuntimeError("v34b exact v401 manifest semantics changed")
    candidate = _read_jsonl_train_only(CANDIDATE_PATH)
    production = _read_jsonl_train_only(PRODUCTION_PATH)
    if len(candidate) != CANDIDATE_ROWS or len(production) != PRODUCTION_ROWS:
        raise RuntimeError("v34b bound train row counts changed")
    return candidate, production


def _panel_order(name: str, items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: canonical_sha256({
            "schema": "eggroll-es-v33a-panel-order",
            "seed": PANEL_SEED,
            "panel": name,
            "unit_id": item["unit_id"],
        }),
    )


def build_runtime_manifest_v34b() -> dict:
    candidate, production = load_bound_rows()
    # This exact implementation is hash-bound above and is the V33 joint-unit prior.
    joint = hardened_v33.build_joint_units(candidate, production)
    paired = joint.pop("paired_units")
    strata = Counter(unit["stratum"] for unit in paired)
    if (
        joint["joint_component_count"] != EXPECTED_JOINT_COMPONENTS
        or len(paired) != EXPECTED_PAIRED_UNITS
        or joint["candidate_only_unit_count"] != EXPECTED_CANDIDATE_ONLY_UNITS
        or joint["production_only_unit_count"] != EXPECTED_PRODUCTION_ONLY_UNITS
        or dict(strata) != EXPECTED_PAIRED_STRATA
        or any(strata[name] < count for name, count in REQUIRED_PAIRED_STRATA.items())
    ):
        raise RuntimeError("v34b paired capacity changed or is insufficient")
    by_stratum = {
        name: sorted(
            (unit for unit in paired if unit["stratum"] == name),
            key=lambda unit: canonical_sha256({
                "schema": "eggroll-es-v33a-stratum-permutation",
                "seed": PANEL_SEED,
                "stratum": name,
                "unit_id": unit["unit_id"],
            }),
        )
        for name in sampler_v13.STRATA
    }
    panels = []
    selected_ids = []
    for panel_index, panel_name in enumerate(PANEL_NAMES):
        selected = []
        for stratum in sampler_v13.STRATA:
            quota = STRATUM_QUOTAS[stratum]
            population = len(by_stratum[stratum])
            start = panel_index * quota
            for unit in by_stratum[stratum][start:start + quota]:
                selected.append({
                    **unit,
                    "inclusion_probability_per_panel": quota / population,
                    "horvitz_thompson_unit_weight": population / quota,
                })
        selected = _panel_order(panel_name, selected)
        if len(selected) != PANEL_SIZE:
            raise RuntimeError("v34b paired panel size changed")
        selected_ids.extend(item["unit_id"] for item in selected)
        panels.append({
            "name": panel_name,
            "role": "optimization" if panel_name in OPTIMIZATION_PANELS else "train_only_screen",
            "rows_per_side_per_direction": PANEL_SIZE,
            "stratum_counts": dict(STRATUM_QUOTAS),
            "ordered_unit_identity_sha256": canonical_sha256(
                [item["unit_id"] for item in selected]
            ),
            "ordered_side_row_identity_sha256": {
                side: canonical_sha256([
                    item["sides"][side]["row_sha256"] for item in selected
                ])
                for side in ("candidate", "production")
            },
            "items": selected,
        })
    if len(selected_ids) != 195 or len(set(selected_ids)) != 195:
        raise RuntimeError("v34b panels are not globally disjoint")
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-runtime-frame-v34b",
        "status": "transient_train_only_frame_no_gpu_or_update_authorization",
        "inputs": {
            "candidate_v401": {
                "path": str(CANDIDATE_PATH),
                "rows": CANDIDATE_ROWS,
                "file_sha256": CANDIDATE_SHA256,
                "manifest_path": str(CANDIDATE_MANIFEST_PATH),
                "manifest_file_sha256": CANDIDATE_MANIFEST_SHA256,
                "manifest_content_sha256": CANDIDATE_MANIFEST_CONTENT_SHA256,
                "freeze_commit": CANDIDATE_FREEZE_COMMIT,
                "builder_file_sha256": CANDIDATE_BUILDER_SHA256,
                "builder_test_file_sha256": CANDIDATE_TEST_SHA256,
            },
            "production": {
                "path": str(PRODUCTION_PATH),
                "rows": PRODUCTION_ROWS,
                "file_sha256": PRODUCTION_SHA256,
                "source_commit": PRODUCTION_SOURCE_COMMIT,
            },
            "joint_mechanics": {
                "implementation": "hash_bound_v33a_exact_joint_units_and_panel_order",
                "builder_file_sha256": HARDENED_V33_BUILDER_SHA256,
                "sampler_file_sha256": SAMPLER_SHA256,
            },
        },
        "joint_frame": {
            **joint,
            "paired_unit_count": len(paired),
            "paired_stratum_counts": dict(strata),
            "selected_paired_unit_count": len(selected_ids),
            "reserve_paired_unit_count": len(paired) - len(selected_ids),
            "selected_unit_id_root_sha256": hashlib.sha256(
                ("\n".join(sorted(selected_ids)) + "\n").encode()
            ).hexdigest(),
        },
        "panel_contract": {
            "panel_seed": PANEL_SEED,
            "representative_seed": REPRESENTATIVE_SEED,
            "panel_names": list(PANEL_NAMES),
            "optimization_panels": list(OPTIMIZATION_PANELS),
            "train_only_screens": list(TRAIN_SCREENS),
            "panel_size": PANEL_SIZE,
            "stratum_quotas": dict(STRATUM_QUOTAS),
            "globally_disjoint_joint_conflict_units": True,
            "fixed_side_representative_every_direction_and_sign": True,
            "same_resident_perturbation_scores_both_sources": True,
            "estimand": "equal_weight_joint_unit_mean_with_exact_stratum_HT_weights",
        },
        "panels": panels,
        "separation": {
            "contains_row_prompt_or_answer_content": False,
            "contains_validation_ood_heldout_or_benchmark_content": False,
            "runtime_details_are_transient": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_runtime_manifest_v34b(value, require_frozen_hashes=False)
    return value


def validate_runtime_manifest_v34b(value, *, require_frozen_hashes=True):
    joint = value.get("joint_frame", {})
    items = [item for panel in value.get("panels", []) for item in panel.get("items", [])]
    if (
        value.get("schema") != "eggroll-es-v401-replacement-fraction-runtime-frame-v34b"
        or value.get("content_sha256_before_self_field") != canonical_sha256(_without_self(value))
        or joint.get("joint_component_count") != EXPECTED_JOINT_COMPONENTS
        or joint.get("paired_unit_count") != EXPECTED_PAIRED_UNITS
        or joint.get("paired_stratum_counts") != EXPECTED_PAIRED_STRATA
        or joint.get("selected_paired_unit_count") != 195
        or joint.get("reserve_paired_unit_count") != 10
        or len(items) != 195
        or len({item.get("unit_id") for item in items}) != 195
        or any(set(item.get("sides", {})) != {"candidate", "production"} for item in items)
        or any(
            not math.isclose(
                item["inclusion_probability_per_panel"] * item["horvitz_thompson_unit_weight"],
                1.0,
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            for item in items
        )
    ):
        raise RuntimeError("v34b transient frame contract changed")
    if require_frozen_hashes and (
        joint.get("paired_assignment_root_sha256") != EXPECTED_PAIRED_ASSIGNMENT_ROOT_SHA256
        or joint.get("selected_unit_id_root_sha256") != EXPECTED_SELECTED_UNIT_ROOT_SHA256
        or value["content_sha256_before_self_field"] != EXPECTED_RUNTIME_FRAME_CONTENT_SHA256
    ):
        raise RuntimeError("v34b transient frame identity changed")
    return value


def aggregate_from_runtime_v34b(runtime: dict) -> dict:
    validate_runtime_manifest_v34b(runtime)
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-frame-v34b",
        "status": "sealed_content_free_train_only_frame_no_gpu_or_update_authorization",
        "inputs": runtime["inputs"],
        "joint_frame": runtime["joint_frame"],
        "panel_contract": runtime["panel_contract"],
        "panels": [
            {key: panel[key] for key in (
                "name", "role", "rows_per_side_per_direction", "stratum_counts",
                "ordered_unit_identity_sha256", "ordered_side_row_identity_sha256",
            )}
            for panel in runtime["panels"]
        ],
        "runtime_frame_content_sha256": runtime["content_sha256_before_self_field"],
        "separation": {
            **runtime["separation"],
            "contains_per_unit_ids_hashes_indices_documents_strata_weights_or_anchors": False,
            "runtime_unit_details_persisted": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def build_manifest() -> dict:
    return aggregate_from_runtime_v34b(build_runtime_manifest_v34b())


def validate_manifest(value: dict) -> dict:
    if (
        value.get("schema") != "eggroll-es-v401-replacement-fraction-frame-v34b"
        or value.get("content_sha256_before_self_field") != canonical_sha256(_without_self(value))
        or value.get("content_sha256_before_self_field")
        != EXPECTED_AGGREGATE_CONTENT_SHA256
        or value.get("separation", {}).get(
            "contains_per_unit_ids_hashes_indices_documents_strata_weights_or_anchors"
        ) is not False
        or value != build_manifest()
    ):
        raise RuntimeError("v34b sealed frame contract changed")
    return value


def exclusive_write(path: Path, value: dict) -> None:
    path = Path(path).resolve()
    if path != OUTPUT_PATH:
        raise ValueError("v34b frame output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v34b frame already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_manifest()
    if not args.dry_run:
        exclusive_write(args.output, value)
    result = {
        "schema": "eggroll-es-v401-replacement-fraction-frame-build-v34b",
        "content_sha256": value["content_sha256_before_self_field"],
        "runtime_frame_content_sha256": value["runtime_frame_content_sha256"],
        "paired_units": value["joint_frame"]["paired_unit_count"],
        "selected_units": value["joint_frame"]["selected_paired_unit_count"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
