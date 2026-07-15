#!/usr/bin/env python3
"""Preregister the train-only paired production/v364 compatibility A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

import build_eggroll_es_joint_panels_v25a as frame_v25a


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DATA_COMPAT_V25A_PREREGISTRATION.json"
)
FRAME_PATH = frame_v25a.OUTPUT_PATH
V17_NEGATIVE_PATH = ROOT / (
    "experiments/eggroll_es_hpo/S6_V17A_PAIRED_DATA_COMPAT_NEGATIVE_EVIDENCE.json"
)
LAYER_PLAN_PATH = ROOT / "experiments/layer_plans/v23a_base_middle_late_dense.json"

FRAME_FILE_SHA256 = "5b7e8f5d24b00e9e6d7a3490e46a134fe95c3beda5bffa9e06b3dc02bcc7f79e"
FRAME_CONTENT_SHA256 = "90b0ecf8d483959076ed87dfd46703e3894ab97cd59f9a1d6fff9bdc79e73b0b"
FRAME_BUILDER_SHA256 = "eb08498f8905f4be62aefc87d080f4c420036211ec0e47858f6a000b39d5c7e8"
FRAME_TEST_SHA256 = "17ab142cb20759b78d6458dc0ca9a7347820773ef0f0ab883aae201c17db551e"
V17_NEGATIVE_FILE_SHA256 = "1fb9a855eccc62d3faa61cfb7756f957728f46099eb1c93ebfacb480bddb3c99"
V17_NEGATIVE_CONTENT_SHA256 = "3566fb66bed9ca694877efad281b451b73c28cc6c0b8b50b7472d00e92f3a3d7"
LAYER_PLAN_FILE_SHA256 = "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747"
LAYER_PLAN_CONTENT_SHA256 = "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f"

EXPERIMENT_NAME = (
    "snapshot_v364_data_v25a_paired_production_candidate_5x39_"
    "alpha_zero_middle_late_basis20260907"
)
PERTURBATION_BASIS_SEED = 20260907
BOOTSTRAP_SEED = 20260908
POPULATION_SIZE = 32
BOOTSTRAP_REPETITIONS = 50_000
FAMILYWISE_ALPHA = 0.05
METRIC_FAMILIES = (
    "optimization_pairwise_cosine",
    "optimization_pairwise_sign_agreement",
    "aggregate_to_optimization_cosine",
    "aggregate_to_optimization_sign_agreement",
    "train_screen_cosine",
    "train_screen_sign_agreement",
)
ENDPOINTS = tuple(
    f"{family}_{summary}"
    for family in METRIC_FAMILIES
    for summary in ("median", "worst")
)
PRIOR_BASIS_HASHES = {
    "v13_v14_v16_v17_v18": "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11",
    "v19a": "d4e46d7d51d5c82cfc981dad3b33db8a1766c70ad570ef931b12550d1bc7bf6c",
    "v20a": "b6d667c2f125f9d0be4d74ef536af03546fecb6c03f2838679f5a315a1ec9852",
    "v21a": "65970861cd06b53e52cf848b2c8b8961160bf9c68f6b1b9f4935a88ba8d314d2",
    "v22a": "f68624388ac0549ac82ba3d1e64a317233c42f900502a6f5c6d6f07071b4c60e",
    "v23a": "aad4ac2e82b55b13fc7a1019b89425d164e7ac8d0e6a8e4fd23c4bcc3f0757eb",
    "v24a": "094dffe3846ad66dc5ea417995aec6cbf210dd6728d47e8c16452d543a48dd1f",
}
LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def perturbation_seeds():
    return [
        int(value) for value in np.random.default_rng(PERTURBATION_BASIS_SEED).integers(
            0, 2**30, size=POPULATION_SIZE, dtype=np.int64,
        )
    ]


def signed_population_schedule():
    seeds = perturbation_seeds()
    schedule = []
    signed_index = 0
    for population_wave in range(8):
        wave_seeds = seeds[population_wave * 4:(population_wave + 1) * 4]
        for sign, negate in (("plus", False), ("minus", True)):
            schedule.append({
                "signed_wave_index": signed_index,
                "population_wave_index": population_wave,
                "sign": sign,
                "negate": negate,
                "engine_direction_indices": list(range(
                    population_wave * 4, (population_wave + 1) * 4,
                )),
                "engine_direction_seeds": wave_seeds,
                "all_four_engines_score_both_data_versions": True,
            })
            signed_index += 1
    return schedule


def bootstrap_draw_plan_sha256():
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    digest = hashlib.sha256()
    for panel in frame_v25a.PANEL_NAMES:
        for stratum in frame_v25a.sampler_v13.STRATA:
            count = frame_v25a.STRATUM_QUOTAS[stratum]
            draw = generator.integers(
                0, count, size=(BOOTSTRAP_REPETITIONS, count), dtype=np.int64,
            )
            header = json.dumps({
                "panel": panel,
                "stratum": stratum,
                "shape": list(draw.shape),
                "dtype": "int64",
            }, sort_keys=True, separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(draw.tobytes(order="C"))
    return digest.hexdigest()


def load_bound_evidence():
    expected = {
        FRAME_PATH: FRAME_FILE_SHA256,
        Path(frame_v25a.__file__).resolve(): FRAME_BUILDER_SHA256,
        ROOT / "test_build_eggroll_es_joint_panels_v25a.py": FRAME_TEST_SHA256,
        V17_NEGATIVE_PATH: V17_NEGATIVE_FILE_SHA256,
        LAYER_PLAN_PATH: LAYER_PLAN_FILE_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v25a bound train-only evidence changed")
    frame = json.loads(FRAME_PATH.read_text())
    negative = json.loads(V17_NEGATIVE_PATH.read_text())
    layer_plan = json.loads(LAYER_PLAN_PATH.read_text())
    frame_v25a.validate_manifest(frame)
    if (
        frame.get("content_sha256_before_self_field") != FRAME_CONTENT_SHA256
        or negative.get("schema")
        != "eggroll-es-paired-data-compat-negative-evidence-v17a"
        or negative.get("content_sha256_before_self_field")
        != V17_NEGATIVE_CONTENT_SHA256
        or canonical_sha256(without_self(negative))
        != V17_NEGATIVE_CONTENT_SHA256
        or negative.get("contains_response_vectors_or_row_content") is not False
        or negative.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
        or negative.get("decision", {}).get("retain_dataset") != "production"
        or layer_plan.get("plan_sha256") != LAYER_PLAN_CONTENT_SHA256
        or layer_plan.get("layers") != [20, 21, 22, 23]
    ):
        raise RuntimeError("v25a aggregate evidence semantics changed")
    return frame, negative, layer_plan


def validate_sha_fields(value):
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if (
                    isinstance(key, str)
                    and "sha256" in key
                    and (
                        not isinstance(item, str)
                        or LOWERCASE_SHA256.fullmatch(item) is None
                    )
                ):
                    raise RuntimeError("v25a persisted SHA field is invalid")
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    return True


def build_preregistration():
    frame, negative, layer_plan = load_bound_evidence()
    seeds = perturbation_seeds()
    schedule = signed_population_schedule()
    basis = {
        "basis_seed": PERTURBATION_BASIS_SEED,
        "generator": (
            "numpy.default_rng(basis_seed).integers(0,2**30,size=32,dtype=int64)"
        ),
        "population_size": POPULATION_SIZE,
        "direction_seeds": seeds,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "signed_population_schedule": schedule,
        "signed_population_schedule_sha256": canonical_sha256(schedule),
    }
    basis["basis_content_sha256"] = canonical_sha256(basis)
    if basis["basis_content_sha256"] in set(PRIOR_BASIS_HASHES.values()):
        raise RuntimeError("v25a fresh basis collides with prior basis")
    value = {
        "schema": "eggroll-es-paired-data-compat-preregistration-v25a",
        "status": "preregistered_runtime_not_yet_authorized",
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_count": 1,
        "hypothesis": (
            "on the frozen common joint-conflict frame, v364 has train-only ES "
            "estimator stability noninferior to production under paired "
            "alpha-zero mechanics"
        ),
        "strict_train_only": {
            "selection_surface": "paired_train_panels_only",
            "validation_opened": False,
            "ood_opened": False,
            "heldout_opened": False,
            "benchmark_opened": False,
            "model_update_or_checkpoint_write": False,
            "dataset_promotion": False,
        },
        "inputs": {
            "candidate_freeze_commit": "de0d5518f5cffe2ee71d8fc6884e506f3c1f3272",
            "candidate": frame["inputs"]["candidate"],
            "production": frame["inputs"]["production"],
            "joint_frame": {
                "path": str(FRAME_PATH),
                "file_sha256": FRAME_FILE_SHA256,
                "content_sha256": FRAME_CONTENT_SHA256,
                "paired_units": frame["joint_frame"]["paired_unit_count"],
                "selected_units": frame["joint_frame"][
                    "selected_paired_unit_count"
                ],
                "reserve_units": frame["joint_frame"][
                    "reserve_paired_unit_count"
                ],
                "paired_strata": frame["joint_frame"]["paired_stratum_counts"],
            },
            "v17_negative_evidence": {
                "path": str(V17_NEGATIVE_PATH),
                "file_sha256": V17_NEGATIVE_FILE_SHA256,
                "content_sha256": V17_NEGATIVE_CONTENT_SHA256,
                "v283_not_reused": True,
            },
        },
        "frozen_recipe": {
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "layer_plan": {
                "path": str(LAYER_PLAN_PATH),
                "file_sha256": LAYER_PLAN_FILE_SHA256,
                "plan_sha256": LAYER_PLAN_CONTENT_SHA256,
                "layers": layer_plan["layers"],
            },
            "sigma": 0.0003,
            "alpha": 0.0,
            "model_update_allowed": False,
            "same_resident_perturbation_scores_both_versions": True,
            "alternate_version_order_by_signed_wave": True,
            "same_fixed_ordered_side_batch_every_direction_and_sign": True,
            "perturbation_basis": basis,
        },
        "panels": {
            "names": list(frame_v25a.PANEL_NAMES),
            "optimization": list(frame_v25a.OPTIMIZATION_PANELS),
            "train_only_screens": list(frame_v25a.TRAIN_SCREENS),
            "panel_size_per_side": frame_v25a.PANEL_SIZE,
            "stratum_quotas": frame_v25a.STRATUM_QUOTAS,
            "globally_disjoint_joint_units": True,
            "horvitz_thompson_weights": True,
            "same_joint_unit_both_versions": True,
            "same_document_preferred_otherwise_same_conflict_component": True,
        },
        "hardware": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "all_four_engines_required_every_signed_wave": True,
            "population_waves": 8,
            "signed_waves": 16,
            "partial_waves_allowed": False,
            "requests_per_engine_per_signed_wave": 390,
            "requests_all_engines_per_signed_wave": 1560,
            "perturbed_requests_all_engines": 24960,
        },
        "analysis": {
            "metric_families": list(METRIC_FAMILIES),
            "endpoints": list(ENDPOINTS),
            "endpoint_count": len(ENDPOINTS),
            "candidate_minus_production_noninferiority_margin": 0.0,
            "all_endpoints_conjunctive": True,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "repetitions": BOOTSTRAP_REPETITIONS,
                "resampling_unit": "paired_joint_unit_within_panel_stratum",
                "same_draw_indices_both_versions": True,
                "draw_order": (
                    "panel_names_then_sampler_v13_strata_each_as_"
                    "rng.integers(0,stratum_count,size=(50000,stratum_count),int64)"
                ),
                "draw_plan_sha256": bootstrap_draw_plan_sha256(),
                "familywise_alpha": FAMILYWISE_ALPHA,
                "multiplicity": "Bonferroni_over_12_endpoints",
                "one_sided_quantile": FAMILYWISE_ALPHA / len(ENDPOINTS),
                "raw_draws_or_replicates_persisted": False,
            },
        },
        "promotion_gate": {
            "pass": (
                "every one-sided familywise bootstrap LCB for candidate-minus-"
                "production median and worst stability endpoints is >= 0"
            ),
            "failure_decision": "retain_production_dataset_and_v13_recipe",
            "pass_authority": (
                "authorize_only_separate_full-v364 train-only HPO preregistration"
            ),
            "pass_does_not_authorize_dataset_promotion_update_or_evaluation": True,
        },
        "integrity": {
            "fresh_o_excl_attempt_and_run_required": True,
            "committed_source_bundle_required": True,
            "exact_reference_restore_after_each_sign": True,
            "population_boundary_selected_and_unselected_audit": True,
            "matched_full_context_a_b_before_and_a_c_after": True,
            "persist_response_vectors_rows_bootstrap_draws_or_replicates": False,
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "runtime_launch_authorized": False,
            "must_bind_this_file_and_content_hash": True,
            "must_reject_validation_ood_heldout_benchmark_and_update_surfaces": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_sha_fields(value)
    return value


def exclusive_write(path, value):
    path = Path(path).resolve()
    if path != PREREGISTRATION_PATH.resolve():
        raise ValueError("v25a preregistration output path changed")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError("v25a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PREREGISTRATION_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration()
    if not args.dry_run:
        exclusive_write(args.output, value)
    print(json.dumps({
        "schema": "eggroll-es-paired-data-compat-preregistration-build-v25a",
        "content_sha256": value["content_sha256_before_self_field"],
        "direction_seed_list_sha256": value["frozen_recipe"][
            "perturbation_basis"
        ]["direction_seed_list_sha256"],
        "signed_population_schedule_sha256": value["frozen_recipe"][
            "perturbation_basis"
        ]["signed_population_schedule_sha256"],
        "bootstrap_draw_plan_sha256": value["analysis"]["bootstrap"][
            "draw_plan_sha256"
        ],
        "gpu_launched": False,
        "runtime_launch_authorized": False,
    }, indent=2, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
