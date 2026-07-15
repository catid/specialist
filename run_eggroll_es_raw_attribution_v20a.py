#!/usr/bin/env python3
"""Authoritative raw-only V20A nested-tier train attribution runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import build_eggroll_es_v20a_union_failure_evidence as failure_v20a
import run_eggroll_es_union_equivalence_v20a as equivalence_v20a


ROOT = Path(__file__).resolve().parent
mechanics_v20a = equivalence_v20a.mechanics_v20a
prereg_v20a = equivalence_v20a.prereg_v20a
frame_v20a = equivalence_v20a.frame_v20a
anchor_v13 = equivalence_v20a.anchor_v13
anchor_v11 = equivalence_v20a.anchor_v11
base = equivalence_v20a.base

FROZEN_MODEL_V20A = equivalence_v20a.FROZEN_MODEL_V20A
FROZEN_OUTPUT_DIRECTORY_V20A = equivalence_v20a.FROZEN_OUTPUT_DIRECTORY_V20A
EXPERIMENT_NAME_V20A = prereg_v20a.EXPERIMENT_NAME_V20A + "_authoritative_raw_attribution"
REPORT_NAME_V20A = "raw_nested_tier_attribution_v20a.json"
TEST_PATH_V20A = (ROOT / "test_run_eggroll_es_raw_attribution_v20a.py").resolve()

SEALED_EQUIVALENCE_COMMIT_V20A = "5f744ec9e9175b68d33dba671f3cd58df994f7ec"
SEALED_FAILURE_EVIDENCE_COMMIT_V20A = (
    "a276bc8b982a04a608535a7c9ccda7e9dc77913f"
)
EQUIVALENCE_PATHS_V20A = {
    "union_equivalence_driver_v20a": Path(equivalence_v20a.__file__).resolve(),
    "union_equivalence_tests_v20a": equivalence_v20a.TEST_PATH_V20A,
}
EQUIVALENCE_HASHES_V20A = {
    "union_equivalence_driver_v20a": (
        "bbd5cdad798726362a43a943f94100ee4aca6fe89c0f245c7db9d21c3e2d2e14"
    ),
    "union_equivalence_tests_v20a": (
        "5c80b14849bc3c3dfc983c1664430e57b80e45f20b20206c0ac95bcdac148260"
    ),
}
FAILURE_PATHS_V20A = {
    "union_failure_builder_v20a": Path(failure_v20a.__file__).resolve(),
    "union_failure_tests_v20a": (
        ROOT / "test_build_eggroll_es_v20a_union_failure_evidence.py"
    ).resolve(),
    "union_failure_evidence_v20a": failure_v20a.OUTPUT_PATH_V20A,
}
FAILURE_HASHES_V20A = {
    "union_failure_builder_v20a": (
        "09473e70a09d1a424247354bc33d46bf540d60f04fae1b5133762d057e27fa68"
    ),
    "union_failure_tests_v20a": (
        "a0cf6911c8a66846cb801b5eef0e04a117df478dd3fc912d632800ce44e053c0"
    ),
    "union_failure_evidence_v20a": (
        "d84fbd7b373b284f3c6a2ffbf1bd52431818ec4e6c60b4f3b320d36b581c9021"
    ),
}
UNION_FAILURE_CONTENT_SHA256_V20A = (
    "2caa4e1658ae794a3193ef28bb3393b966f1d329d574bc8d0b2534e3fb1f3302"
)
FORBIDDEN_SURFACE_TOKENS_V20A = equivalence_v20a.FORBIDDEN_SURFACE_TOKENS_V20A
FORBIDDEN_PERSISTED_KEYS_V20A = (
    equivalence_v20a.FORBIDDEN_PERSISTED_KEYS_V20A
)
MOE_OVERRIDE_ENVIRONMENT_V20A = equivalence_v20a.MOE_OVERRIDE_ENVIRONMENT_V20A

canonical_sha256 = equivalence_v20a.canonical_sha256
file_sha256 = equivalence_v20a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file(path, commit, digest):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v20a raw committed artifact changed: {relative}")


def _verify_raw_bindings_v20a():
    for paths, hashes, commit in (
        (EQUIVALENCE_PATHS_V20A, EQUIVALENCE_HASHES_V20A, SEALED_EQUIVALENCE_COMMIT_V20A),
        (FAILURE_PATHS_V20A, FAILURE_HASHES_V20A, SEALED_FAILURE_EVIDENCE_COMMIT_V20A),
    ):
        for key, path in paths.items():
            _verify_commit_file(path, commit, hashes[key])
    evidence = json.loads(failure_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8"))
    failure_v20a.validate_evidence_v20a(evidence)
    if (
        evidence["content_sha256_before_self_field"]
        != UNION_FAILURE_CONTENT_SHA256_V20A
        or evidence["decision"]["union_scoring_authorized_for_v20a"] is not False
        or evidence["decision"]["raw_arm_scoring_remains_authoritative"] is not True
    ):
        raise RuntimeError("v20a raw runtime union failure binding changed")
    return evidence


def implementation_identity_v20a():
    inherited = equivalence_v20a.implementation_identity_v20a()
    _verify_raw_bindings_v20a()
    paths = {
        **EQUIVALENCE_PATHS_V20A,
        **FAILURE_PATHS_V20A,
        "raw_runtime_driver_v20a": Path(__file__).resolve(),
        "raw_runtime_tests_v20a": TEST_PATH_V20A,
    }
    files = {
        key: {"path": str(path), "file_sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    return {
        "files": files,
        "inherited_equivalence_bundle_sha256": inherited["bundle_sha256"],
        "failure_evidence_bundle_sha256": canonical_sha256({
            key: files[key] for key in FAILURE_PATHS_V20A
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def _source_provenance_v20a(implementation):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        raw = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v20a raw source differs from HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    value = {
        "schema": "eggroll-es-raw-attribution-source-bundle-v20a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def resident_signed_wave_schedule_v20a(seeds=None):
    seeds = list(prereg_v20a.PERTURBATION_SEEDS_V20A if seeds is None else seeds)
    if seeds != prereg_v20a.PERTURBATION_SEEDS_V20A:
        raise RuntimeError("v20a raw exact perturbation basis changed")
    orders = prereg_v20a.signed_wave_arm_orders_v20a()
    schedule = []
    for wave_index, start in enumerate(range(0, 32, 4)):
        wave = [int(seed) for seed in seeds[start:start + 4]]
        for sign_index, sign in enumerate(("plus", "minus")):
            index = 2 * wave_index + sign_index
            schedule.append({
                "signed_wave_index": index,
                "population_wave_index": wave_index,
                "sign": sign,
                "negate": sign == "minus",
                "engine_seeds": wave,
                "resident_arm_order": orders[index]["arm_order"],
                "restore_after_all_four_arms": True,
            })
    if len(schedule) != 16 or any(
        set(item["resident_arm_order"]) != set(mechanics_v20a.ARMS_V20A)
        for item in schedule
    ):
        raise RuntimeError("v20a raw signed-wave schedule changed")
    return schedule


def _load_bound_inputs_v20a(layer_bundle):
    preregistration, panels = equivalence_v20a._load_bound_inputs_v20a(layer_bundle)
    evidence = _verify_raw_bindings_v20a()
    return preregistration, panels, evidence


def recipe_v20a(layer_bundle, preregistration, panels, evidence, implementation, moe):
    if moe != equivalence_v20a._declared_moe_backend_v20a():
        raise RuntimeError("v20a raw default Triton declaration changed")
    schedule = resident_signed_wave_schedule_v20a()
    value = {
        "schema": "eggroll-es-authoritative-raw-attribution-recipe-v20a",
        "experiment_name": EXPERIMENT_NAME_V20A,
        "model": str(FROZEN_MODEL_V20A),
        "layer_plan": copy.deepcopy(preregistration["frozen_recipe"]["layer_plan"]),
        "layers": [20, 21, 22, 23],
        "sigma": 0.0003,
        "alpha": 0.0,
        "population_size": 32,
        "perturbation_basis_sha256": prereg_v20a.PERTURBATION_BASIS_SHA256_V20A,
        "perturbation_seed_list_sha256": prereg_v20a.PERTURBATION_SEED_LIST_SHA256_V20A,
        "signed_wave_schedule_sha256": canonical_sha256(schedule),
        "signed_wave_count": 16,
        "score_all_four_arms_before_restore": True,
        "restore_once_after_all_four_arms": True,
        "authoritative_raw_scoring": {
            "requests_per_engine_per_signed_wave": 1020,
            "requests_by_arm": {
                "production_only": 240,
                "patch_tier_2_only": 250,
                "patch_tiers_2_3": 260,
                "patch_all_tiers": 270,
            },
            "dense_result_commitment_count": 2560,
        },
        "union_scoring": {
            "authorized": False,
            # The sealed equivalence implementation remains identity-bound, but
            # this runtime never executes its union planner or scoring path.
            "planner_called_by_runtime": False,
            "failure_evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
        },
        "panel_bundle_content_sha256": panels["content_sha256_before_self_field"],
        "draw_plan_content_sha256": mechanics_v20a.DRAW_PLAN_CONTENT_SHA256_V20A,
        "analysis": {
            "contrast_count": 5,
            "endpoints_per_contrast": 12,
            "hypothesis_count": 60,
            "bootstrap_repetitions": 50_000,
            "familywise_one_sided_quantile": 0.05 / 60,
        },
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "hardware": {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_every_signed_wave": True,
        },
        "moe_backend": copy.deepcopy(moe),
        "authority": {
            "train_only_raw_attribution_runtime": True,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
        },
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V20A),
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _parser_v20a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20a-raw-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_v20a(args, layer, implementation, recipe):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer)
    if (
        recipe["content_sha256_before_self_field"]
        != canonical_sha256(_without_self(recipe))
        or recipe["alpha"] != 0.0
        or recipe["signed_wave_count"] != 16
        or recipe["authoritative_raw_scoring"]["requests_per_engine_per_signed_wave"]
        != 1020
        or recipe["authoritative_raw_scoring"]["dense_result_commitment_count"]
        != 2560
        or recipe["union_scoring"]["authorized"] is not False
        or recipe["union_scoring"]["planner_called_by_runtime"] is not False
        or any(recipe["authority"][key] is not False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_allowed", "dataset_promotion_allowed",
        ))
    ):
        raise ValueError("v20a raw frozen recipe changed")
    if not args.v20a_raw_dry_run and (
        args.expected_implementation_bundle_sha256 is None
        or args.expected_recipe_sha256 is None
    ):
        raise ValueError("v20a raw launch requires exact implementation and recipe hashes")
    if args.expected_implementation_bundle_sha256 not in (
        None, implementation["bundle_sha256"]
    ):
        raise ValueError("v20a raw implementation bundle hash changed")
    if args.expected_recipe_sha256 not in (
        None, recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v20a raw recipe hash changed")


class RawNestedTierAttributionRuntimeMixinV20A(
    equivalence_v20a.UnionScoringEquivalenceRuntimeMixinV20A
):
    def configure_raw_attribution_v20a(self, panel_bundle, *, frozen_layer_plan):
        config = (
            equivalence_v20a.UnionScoringEquivalenceRuntimeMixinV20A
            .configure_union_equivalence_v20a(
                self, panel_bundle, frozen_layer_plan=frozen_layer_plan
            )
        )
        self._v20a_panel_bundle = mechanics_v20a.validate_panel_bundle_v20a(
            copy.deepcopy(panel_bundle)
        )
        config.update({
            "schema": "eggroll-es-raw-attribution-runtime-configuration-v20a",
            "union_scoring_authorized": False,
            "union_planner_called": False,
            "train_only_attribution_runtime_opened": True,
        })
        config.pop("attribution_runtime_opened", None)
        return config

    def _raw_reference_probe_v20a(self, prepared):
        scored = self._score_generated_v20a(self._generate_raw_v20a(prepared))
        result = canonical_sha256([
            {
                "key": list(key),
                "dense_sha256": dense_hash,
                "score_bytes_sha256": hashlib.sha256(
                    np.asarray(scores, dtype=np.float64).tobytes()
                ).hexdigest(),
            }
            for key, (scores, dense_hash) in sorted(scored.items())
        ])
        scored = None
        return result

    def _score_resident_arm_v20a(
        self, arm, prepared, schedule_item, unit_scores, dense_commitments,
    ):
        items = prepared[arm]["prompt_items"]
        prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in items]
        batches = self._resolve([
            engine.generate.remote(
                list(prompts), self._dense_sampling_params_v4(0), use_tqdm=False
            )
            for engine in self.engines
        ])
        if len(batches) != 4 or any(len(batch) != len(prompts) for batch in batches):
            raise RuntimeError("v20a raw resident arm generation incomplete")
        sign_index = ("plus", "minus").index(schedule_item["sign"])
        for engine_index, outputs in enumerate(batches):
            direction_index = 4 * schedule_item["population_wave_index"] + engine_index
            for panel_index, panel_name in enumerate(mechanics_v20a.PANEL_NAMES_V20A):
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                rewards, dense_hash = equivalence_v20a.driver_v19a._score_panel_unit_outputs_v19a(
                    panel["dense_items"], outputs[start:end], end - start
                )
                unit_scores[arm][panel_index, sign_index, direction_index] = rewards
                dense_commitments.append(dense_hash)
        return canonical_sha256(dense_commitments[-40:])

    def _run_signed_wave_v20a(
        self, schedule_item, prepared, unit_scores, dense_commitments,
    ):
        index = schedule_item.get("signed_wave_index") if isinstance(schedule_item, dict) else None
        schedule = resident_signed_wave_schedule_v20a()
        if not isinstance(index, int) or index not in range(16) or schedule_item != schedule[index]:
            raise RuntimeError("v20a raw signed-wave schedule item changed")
        captures = {}
        try:
            self._perturb_signed_wave_v19a(
                schedule_item["engine_seeds"], schedule_item["negate"]
            )
            for arm in schedule_item["resident_arm_order"]:
                captures[arm] = self._score_resident_arm_v20a(
                    arm, prepared, schedule_item, unit_scores, dense_commitments
                )
        finally:
            restore_hash = self._restore_and_verify_signed_wave_v19a()
        if tuple(captures) != tuple(schedule_item["resident_arm_order"]):
            raise RuntimeError("v20a raw signed-wave transaction changed")
        return restore_hash

    def estimate_raw_attribution_v20a(self, seeds):
        schedule = resident_signed_wave_schedule_v20a(seeds)
        prepared, request_identity = self._prepared_fixed_batches_v20a()
        pre_probe = self._raw_reference_probe_v20a(prepared)
        unit_scores = {
            arm: np.full(
                (10, 2, 32, frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]),
                np.nan,
                dtype=np.float64,
            )
            for arm in mechanics_v20a.ARMS_V20A
        }
        dense_commitments = []
        restore_hashes = []
        for item in schedule:
            restore_hashes.append(self._run_signed_wave_v20a(
                item, prepared, unit_scores, dense_commitments
            ))
        if (
            any(not np.isfinite(values).all() for values in unit_scores.values())
            or len(dense_commitments) != 2560
            or len(restore_hashes) != 16
        ):
            raise RuntimeError("v20a raw population capture incomplete")
        boundary = self._population_boundary_audit_v4(0)
        unselected, unselected_sha = (
            equivalence_v20a.driver_v19a._validate_population_boundary_v19a(
                boundary, self._v4_layer_plan_install
            )
        )
        post_probe = self._raw_reference_probe_v20a(prepared)
        if pre_probe != post_probe:
            raise RuntimeError("v20a raw pre/post reference probe drifted")
        compact = mechanics_v20a.build_compact_estimator_summary_v20a(
            unit_scores, self._v20a_panel_bundle
        )
        unit_scores = None
        runtime_integrity = {
            "all_four_tp1_engines_every_signed_wave": True,
            "all_ten_panels_every_direction_sign_and_arm": True,
            "all_sixteen_signed_waves_complete": True,
            "latin_arm_order_complete": True,
            "exact_reference_restored_once_per_signed_wave": True,
            "pre_post_raw_reference_probes_equal": True,
            "population_boundary_audit_passed": True,
            "unselected_origin_audit_passed": unselected["passed"],
            "union_scoring_called_or_used": False,
            "all_integrity_audits_passed": True,
        }
        summary = {
            "schema": "eggroll-es-raw-nested-tier-attribution-summary-v20a",
            "experiment_name": EXPERIMENT_NAME_V20A,
            "alpha": 0.0,
            "sigma": 0.0003,
            "runtime_integrity": runtime_integrity,
            "arms": compact["arms"],
            "paired_bootstrap": compact["paired_bootstrap"],
            "union_scoring_authorized_or_used": False,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_applied": False,
            "persisted_response_vectors_or_row_content": False,
            "bootstrap_draws_persisted": False,
            "unit_scores_persisted": False,
        }
        summary["content_sha256_before_self_field"] = canonical_sha256(summary)
        gate = mechanics_v20a.evaluate_attribution_gate_v20a(summary)
        audit = {
            "schema": "eggroll-es-raw-attribution-runtime-audit-v20a",
            "fixed_request_identity_sha256": canonical_sha256(request_identity),
            "token_boundary_audit_sha256": self._v20a_token_boundary_audit_sha256,
            "pre_post_probe_identity_sha256": canonical_sha256({
                "pre": pre_probe, "post": post_probe
            }),
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "dense_result_commitments_sha256": canonical_sha256(dense_commitments),
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "unselected_origin_sha256": boundary["unselected_origin_sha256"],
            "unselected_origin_audit_sha256": unselected_sha,
            "signed_wave_count": 16,
            "panel_count": 10,
            "requests_per_engine_per_signed_wave": 1020,
            "dense_result_commitment_count": 2560,
            "union_scoring_called_or_used": False,
            "per_unit_scores_persisted": False,
            "bootstrap_replicates_persisted": False,
            "bootstrap_draws_persisted": False,
            "row_content_persisted": False,
        }
        audit["content_sha256_before_self_field"] = canonical_sha256(audit)
        return summary, gate, audit

    @staticmethod
    def _closed_union_v20a(*_args, **_kwargs):
        raise RuntimeError("v20a raw runtime permanently disables union scoring")

    run_union_equivalence_v20a = _closed_union_v20a
    configure_union_equivalence_v20a = _closed_union_v20a
    _generate_union_v20a = _closed_union_v20a
    _equivalence_state_v20a = _closed_union_v20a
    build_union_request_plan_v20a = _closed_union_v20a


def load_runtime_trainer_v20a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    parent = anchor_v13.load_trainer(layer_bundle)

    class RawNestedTierAttributionRuntimeTrainerV20A(
        RawNestedTierAttributionRuntimeMixinV20A, parent
    ):
        pass
    return RawNestedTierAttributionRuntimeTrainerV20A


def _make_trainer_v20a(layer_bundle):
    trainer_class = load_runtime_trainer_v20a(layer_bundle)
    return trainer_class(
        model_name=str(FROZEN_MODEL_V20A), checkpoint=None, sigma=0.0003,
        alpha=0.0, population_size=32, reward_shaping="z-scores",
        num_iterations=1, max_tokens=1, batch_size=270, mini_batch_size=270,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template, train_dataloader=[],
        eval_dataloader_dict={}, eval_freq=1, n_vllm_engines=4,
        n_gpu_per_vllm_engine=1, logging="none", global_seed=43,
        use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_V20A,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(FROZEN_OUTPUT_DIRECTORY_V20A),
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def validate_compact_report_v20a(report, *, expected_recipe, expected_implementation):
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    audit = report.get("runtime_audit", {}) if isinstance(report, dict) else {}
    configuration = report.get("configuration", {}) if isinstance(report, dict) else {}
    expected_top_level = {
        "schema", "recipe", "configuration", "runtime_audit", "summary",
        "gate", "implementation", "union_scoring_authorized_or_used",
        "model_update_applied", "checkpoint_written",
        "evaluation_surfaces_opened", "dataset_promotion_applied",
        "content_sha256_before_self_field",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_top_level
        or report.get("schema") != "eggroll-es-authoritative-raw-attribution-report-v20a"
        or report.get("recipe") != expected_recipe
        or report.get("implementation") != expected_implementation
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(report))
        or summary.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(summary))
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(audit))
        or report.get("gate") != mechanics_v20a.evaluate_attribution_gate_v20a(summary)
        or report.get("union_scoring_authorized_or_used") is not False
        or summary.get("union_scoring_authorized_or_used") is not False
        or audit.get("union_scoring_called_or_used") is not False
        or audit.get("signed_wave_count") != 16
        or audit.get("requests_per_engine_per_signed_wave") != 1020
        or audit.get("dense_result_commitment_count") != 2560
        or configuration.get("engine_count") != 4
        or configuration.get("tp_per_engine") != 1
        or configuration.get("gpu_ids") != [0, 1, 2, 3]
        or configuration.get("union_scoring_authorized") is not False
        or configuration.get("union_planner_called") is not False
        or configuration.get("train_only_attribution_runtime_opened") is not True
        or any(configuration.get(key) is not False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_surfaces_opened", "dataset_promotion_allowed",
        ))
        or any(report.get(key) is not False for key in (
            "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
        ))
        or any(
            str(key).lower() in FORBIDDEN_PERSISTED_KEYS_V20A
            for key in _recursive_keys(report)
        )
    ):
        raise RuntimeError("v20a raw compact report changed")
    return report


def _claim_paths(attempt):
    root = (FROZEN_OUTPUT_DIRECTORY_V20A / EXPERIMENT_NAME_V20A).resolve()
    for _ in range(32):
        launch = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"
        attempt_path = root / "attempts" / f"{launch}.json"
        run_dir = root / "runs" / launch
        current = copy.deepcopy(attempt)
        current.update({"launch_id": launch, "run_directory": str(run_dir)})
        try:
            equivalence_v20a._exclusive_write_json_v20a(attempt_path, current)
        except FileExistsError:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        return attempt_path, run_dir, current
    raise RuntimeError("v20a raw could not claim fresh paths")


def run_exact_v20a(layer, panels, implementation, recipe):
    attempt = {
        "schema": "eggroll-es-raw-attribution-attempt-v20a",
        "status": "launching", "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V20A, "recipe": recipe,
        "source_provenance": _source_provenance_v20a(implementation),
        "union_scoring_authorized_or_used": False,
        "model_update_applied": False, "checkpoint_written": False,
        "evaluation_surfaces_opened": False, "dataset_promotion_applied": False,
    }
    attempt_path, run_dir, attempt = _claim_paths(attempt)
    report_path = run_dir / REPORT_NAME_V20A
    trainer = None
    failure = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v20a(layer)
        configuration = trainer.configure_raw_attribution_v20a(
            panels, frozen_layer_plan=layer
        )
        summary, gate, audit = trainer.estimate_raw_attribution_v20a(
            prereg_v20a.PERTURBATION_SEEDS_V20A
        )
    except BaseException as error:
        failure = error
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except BaseException as cleanup:
                if failure is None:
                    failure = cleanup
    if failure is not None:
        attempt.update({
            "status": "failed", "phase": "inside_raw_attribution_runtime",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure)
            }),
            "report_exists_after_attempt": report_path.exists(),
        })
        equivalence_v20a._rewrite_json_v20a(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-authoritative-raw-attribution-report-v20a",
        "recipe": recipe, "configuration": configuration,
        "runtime_audit": audit, "summary": summary, "gate": gate,
        "implementation": implementation,
        "union_scoring_authorized_or_used": False,
        "model_update_applied": False, "checkpoint_written": False,
        "evaluation_surfaces_opened": False, "dataset_promotion_applied": False,
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    validate_compact_report_v20a(
        report, expected_recipe=recipe, expected_implementation=implementation
    )
    equivalence_v20a._exclusive_write_json_v20a(report_path, report)
    attempt.update({
        "status": "complete", "phase": "after_cleanup_and_compact_report",
        "report_exists_after_attempt": True,
        "report_binding": {
            "path": str(report_path), "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
    })
    equivalence_v20a._rewrite_json_v20a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    equivalence_v20a._assert_train_only_argv_v20a(argv)
    moe = equivalence_v20a._validate_moe_backend_environment_v20a()
    layer, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v20a().parse_args(remaining)
    implementation = implementation_identity_v20a()
    preregistration, panels, evidence = _load_bound_inputs_v20a(layer)
    recipe = recipe_v20a(
        layer, preregistration, panels, evidence, implementation, moe
    )
    validate_runtime_v20a(args, layer, implementation, recipe)
    if args.v20a_raw_dry_run:
        payload = {
            "schema": "eggroll-es-authoritative-raw-attribution-dry-run-v20a",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "implementation": implementation, "recipe": recipe,
            "gpu_launched": False, "union_scoring_authorized_or_used": False,
        }
        payload["content_sha256_before_self_field"] = canonical_sha256(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v20a(layer, panels, implementation, recipe)


if __name__ == "__main__":
    main()
