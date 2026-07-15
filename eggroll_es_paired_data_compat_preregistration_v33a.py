#!/usr/bin/env python3
"""Preregister the train-only paired production/v364 compatibility A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_joint_panels_v33a as frame_v33a


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DATA_COMPAT_V33A_PREREGISTRATION.json"
)
FRAME_PATH = frame_v33a.OUTPUT_PATH
LAYER_PLAN_PATH = ROOT / "experiments/layer_plans/v23a_base_middle_late_dense.json"
V25A_AGGREGATE_EVIDENCE_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V25A_PAIRED_PROMISING_UNCONFIRMED_EVIDENCE_R1.json"
)

FRAME_FILE_SHA256 = "8c8ab38e03949e01982701b58e0f273c23305e67b114f7c607e7e4d83d10666a"
FRAME_CONTENT_SHA256 = "1690210499f6ac2739b6d63c02c8b3101dc36027438042c9534d2ad9de5c9d68"
FRAME_BUILDER_SHA256 = "129ede4440fd13325edee309bf70d7e0e2356e42dacbac9aed0f4243fe54cbaf"
FRAME_TEST_SHA256 = "269adc5133f42886a9f7bbb3b19d89a04d7fd04c0d00d6b9762ecc8db48cf21c"
LAYER_PLAN_FILE_SHA256 = "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747"
LAYER_PLAN_CONTENT_SHA256 = "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f"
V25A_AGGREGATE_EVIDENCE_COMMIT = "1fb518c2c3eed467eed47fa4a9b498cfdca05c24"
V25A_AGGREGATE_EVIDENCE_FILE_SHA256 = (
    "311966e1b8b03d2354289fde9ef010f47709862d11fdec9e236d7f4de0a9a65c"
)
V25A_AGGREGATE_EVIDENCE_CONTENT_SHA256 = (
    "9675929734e3f2ed53d465c87b73e6b6dec0ed71a958221826f33af158c01cb4"
)

EXPERIMENT_NAME = (
    "snapshot_v364_data_v33a_paired_production_candidate_5x39_"
    "alpha_zero_middle_late_basis20261008"
)
PERTURBATION_BASIS_SEED = 20261008
BOOTSTRAP_SEED = 20261009
POPULATION_SIZE = 64
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
    "v25a": "fc51491433efde545ed17450fc625c8ec38cd3bb1b48c2ac74066c6fec2cba24",
    "v30a": "fb9d939f9fc3444694c74cd6805f24a54a936a532b02cf73e4711abc885145b5",
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
    for population_wave in range(POPULATION_SIZE // 4):
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
    for panel in frame_v33a.PANEL_NAMES:
        for stratum in frame_v33a.sampler_v13.STRATA:
            count = frame_v33a.STRATUM_QUOTAS[stratum]
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
        Path(frame_v33a.__file__).resolve(): FRAME_BUILDER_SHA256,
        ROOT / "test_build_eggroll_es_joint_panels_v33a.py": FRAME_TEST_SHA256,
        LAYER_PLAN_PATH: LAYER_PLAN_FILE_SHA256,
        V25A_AGGREGATE_EVIDENCE_PATH: V25A_AGGREGATE_EVIDENCE_FILE_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v33a bound train-only evidence changed")
    frame = json.loads(FRAME_PATH.read_text())
    layer_plan = json.loads(LAYER_PLAN_PATH.read_text())
    aggregate_evidence = json.loads(
        V25A_AGGREGATE_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    committed_evidence = subprocess.check_output(
        [
            "git", "show",
            f"{V25A_AGGREGATE_EVIDENCE_COMMIT}:"
            f"{V25A_AGGREGATE_EVIDENCE_PATH.relative_to(ROOT).as_posix()}",
        ],
        cwd=ROOT,
    )
    frame_v33a.validate_manifest(frame)
    if (
        frame.get("content_sha256_before_self_field") != FRAME_CONTENT_SHA256
        or layer_plan.get("plan_sha256") != LAYER_PLAN_CONTENT_SHA256
        or layer_plan.get("layers") != [20, 21, 22, 23]
        or hashlib.sha256(committed_evidence).hexdigest()
        != V25A_AGGREGATE_EVIDENCE_FILE_SHA256
        or aggregate_evidence.get("schema")
        != "eggroll-es-v25a-paired-promising-unconfirmed-evidence-r1"
        or aggregate_evidence.get("content_sha256_before_self_field")
        != V25A_AGGREGATE_EVIDENCE_CONTENT_SHA256
        or canonical_sha256(without_self(aggregate_evidence))
        != V25A_AGGREGATE_EVIDENCE_CONTENT_SHA256
        or aggregate_evidence.get("aggregate_result", {}).get("endpoint_count")
        != 12
        or aggregate_evidence.get("aggregate_result", {}).get(
            "observed_candidate_minus_production_nonnegative"
        ) != 12
        or aggregate_evidence.get("aggregate_result", {}).get(
            "observed_candidate_minus_production_positive"
        ) != 11
        or aggregate_evidence.get("aggregate_result", {}).get(
            "familywise_lcbs_nonnegative"
        ) != 0
        or aggregate_evidence.get("decision", {}).get("global_gate_passed")
        is not False
        or aggregate_evidence.get("decision", {}).get(
            "confirmation_authorized_by_this_gate"
        ) is not False
        or aggregate_evidence.get("decision", {}).get("retain_dataset")
        != "production"
        or aggregate_evidence.get("decision", {}).get("retain_recipe") != "v13"
        or aggregate_evidence.get(
            "contains_response_vectors_unit_scores_or_bootstrap_replicates"
        ) is not False
        or aggregate_evidence.get(
            "contains_dataset_rows_questions_answers_or_document_content"
        ) is not False
        or aggregate_evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v33a aggregate evidence semantics changed")
    return frame, aggregate_evidence, layer_plan


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
                    raise RuntimeError("v33a persisted SHA field is invalid")
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    return True


def build_preregistration():
    frame, aggregate_evidence, layer_plan = load_bound_evidence()
    seeds = perturbation_seeds()
    schedule = signed_population_schedule()
    basis = {
        "basis_seed": PERTURBATION_BASIS_SEED,
        "generator": (
            "numpy.default_rng(basis_seed).integers(0,2**30,size=64,dtype=int64)"
        ),
        "population_size": POPULATION_SIZE,
        "direction_seeds": seeds,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "signed_population_schedule": schedule,
        "signed_population_schedule_sha256": canonical_sha256(schedule),
    }
    basis["basis_content_sha256"] = canonical_sha256(basis)
    if basis["basis_content_sha256"] in set(PRIOR_BASIS_HASHES.values()):
        raise RuntimeError("v33a fresh basis collides with prior basis")
    value = {
        "schema": "eggroll-es-paired-data-compat-preregistration-v33a",
        "status": "preregistered_runtime_not_yet_authorized",
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_count": 1,
        "hypothesis": (
            "on the frozen common joint-conflict frame, v364 has train-only ES "
            "estimator stability noninferior to production under paired "
            "alpha-zero mechanics"
        ),
        "audited_prior_aggregate_evidence": {
            "version": "v25a_r1_promising_unconfirmed",
            "path": str(V25A_AGGREGATE_EVIDENCE_PATH),
            "commit": V25A_AGGREGATE_EVIDENCE_COMMIT,
            "file_sha256": V25A_AGGREGATE_EVIDENCE_FILE_SHA256,
            "content_sha256": V25A_AGGREGATE_EVIDENCE_CONTENT_SHA256,
            "v25a_global_gate_passed": False,
            "v25a_confirmation_authorized": False,
            "v25a_structural_fact_only": (
                "11_of_12_point_deltas_positive_1_zero_and_0_of_12_"
                "familywise_lcbs_nonnegative"
            ),
            "v25a_not_reinterpreted_as_pass_or_confirmation": True,
            "all_12_conjunctive_familywise_endpoints_preserved": True,
            "familywise_alpha": 0.05,
            "multiplicity": "Bonferroni_over_12_endpoints",
            "noninferiority_margin": 0.0,
            "optimization_panels": 3,
            "untouched_train_screens": 2,
            "perturbation_or_bootstrap_draw_reuse": False,
            "prior_population_size": 32,
            "new_population_size": 64,
            "bootstrap_repetitions": 50_000,
            "posthoc_power_margin_multiplicity_or_threshold_change": False,
            "candidate_signal_source": "v25a_aggregate_structural_direction_only",
        },
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
            "candidate_freeze_commit": frame_v33a.CANDIDATE_FREEZE_COMMIT,
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
            "v25a_aggregate_evidence": {
                "path": str(V25A_AGGREGATE_EVIDENCE_PATH),
                "commit": V25A_AGGREGATE_EVIDENCE_COMMIT,
                "file_sha256": V25A_AGGREGATE_EVIDENCE_FILE_SHA256,
                "content_sha256": V25A_AGGREGATE_EVIDENCE_CONTENT_SHA256,
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
            "names": list(frame_v33a.PANEL_NAMES),
            "optimization": list(frame_v33a.OPTIMIZATION_PANELS),
            "train_only_screens": list(frame_v33a.TRAIN_SCREENS),
            "panel_size_per_side": frame_v33a.PANEL_SIZE,
            "stratum_quotas": frame_v33a.STRATUM_QUOTAS,
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
            "population_waves": 16,
            "synchronized_four_engine_signed_waves": 32,
            "engine_signed_direction_evaluations": 128,
            "partial_waves_allowed": False,
            "requests_per_engine_per_signed_wave": 390,
            "requests_all_engines_per_signed_wave": 1560,
            "perturbed_requests_all_engines": 49920,
            "full_context_requests_all_engines": 4680,
            "total_generation_requests": 54600,
        },
        "analysis": {
            "metric_families": list(METRIC_FAMILIES),
            "endpoints": list(ENDPOINTS),
            "endpoint_count": len(ENDPOINTS),
            "candidate_minus_production_noninferiority_margin": 0.0,
            "all_endpoints_conjunctive": True,
            "all_12_observed_point_deltas_must_be_nonnegative": True,
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
                "production median and worst stability endpoints is >= 0 and "
                "all 12 observed candidate-minus-production point deltas are >= 0"
            ),
            "failure_decision": "retain_production_dataset_and_v13_recipe",
            "pass_authority": (
                "authorize_only_separate_fresh-basis-v364_confirmation_preregistration"
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
            "persist_per_unit_identity_hash_index_document_stratum_weight_or_anchor": False,
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
        raise ValueError("v33a preregistration output path changed")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError("v33a preregistration already exists") from error
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
        "schema": "eggroll-es-paired-data-compat-preregistration-build-v33a",
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
