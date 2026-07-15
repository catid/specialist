#!/usr/bin/env python3
"""Authoritative raw-only V21A production-v331 compatibility runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run_eggroll_es_disjoint_tier_attribution_v19a as driver_v19a
import train_eggroll_es_production_v331_patch_v21a as mechanics_v21a


ROOT = Path(__file__).resolve().parent
prereg_v21a = mechanics_v21a.prereg_v21a
frame_v21a = mechanics_v21a.frame_v21a
anchor_v13 = driver_v19a.anchor_v13
anchor_v4 = driver_v19a.anchor_v4
base = driver_v19a.base

FROZEN_MODEL_V21A = prereg_v21a.MODEL_PATH_V21A
FROZEN_OUTPUT_DIRECTORY_V21A = driver_v19a.FROZEN_OUTPUT_DIRECTORY_V19A
EXPERIMENT_NAME_V21A = prereg_v21a.EXPERIMENT_NAME_V21A + "_authoritative_raw"
REPORT_NAME_V21A = "raw_production_v331_patch_attribution_v21a.json"
TEST_PATH_V21A = (ROOT / "test_run_eggroll_es_raw_attribution_v21a.py").resolve()

SEALED_PREREG_COMMIT_V21A = "74b9775e019a25f01d860e3508c9979646624db8"
SEALED_MECHANICS_COMMIT_V21A = "d97c552cfb99ffeaa50c1342df132adc41c31f2e"
PREREG_PATHS_V21A = {
    "frame_builder_v21a": Path(frame_v21a.__file__).resolve(),
    "frame_tests_v21a": (
        ROOT / "test_build_eggroll_es_production_v331_patch_frame_v21a.py"
    ).resolve(),
    "frame_certificate_v21a": frame_v21a.OUTPUT_PATH_V21A,
    "prereg_builder_v21a": Path(prereg_v21a.__file__).resolve(),
    "prereg_tests_v21a": (
        ROOT / "test_eggroll_es_production_v331_patch_preregistration_v21a.py"
    ).resolve(),
    "preregistration_v21a": prereg_v21a.OUTPUT_PATH_V21A,
}
PREREG_HASHES_V21A = {
    "frame_builder_v21a": prereg_v21a.FRAME_BUILDER_SHA256_V21A,
    "frame_tests_v21a": prereg_v21a.FRAME_TEST_SHA256_V21A,
    "frame_certificate_v21a": prereg_v21a.FRAME_CERTIFICATE_SHA256_V21A,
    "prereg_builder_v21a": (
        "06ffa648d9140ef2d5723d07bad7e1bb9fbde9684a1db86a0847afdc926c84df"
    ),
    "prereg_tests_v21a": (
        "d46dabc63c90a0c37293f26fdc74cfe38e8b89b970dc8cb42a9fcf14a8cdfa0a"
    ),
    "preregistration_v21a": (
        "d0a9284bb5a944b6f5059f4ed55a4616e2d00acaf2a8bdbba731603eb2126dbd"
    ),
}
MECHANICS_PATHS_V21A = {
    "draw_builder_v21a": Path(mechanics_v21a.draw_v21a.__file__).resolve(),
    "draw_tests_v21a": (
        ROOT / "test_build_eggroll_es_production_v331_patch_draw_plan_v21a.py"
    ).resolve(),
    "draw_plan_v21a": mechanics_v21a.draw_v21a.OUTPUT_PATH_V21A,
    "mechanics_v21a": Path(mechanics_v21a.__file__).resolve(),
    "mechanics_tests_v21a": (
        ROOT / "test_train_eggroll_es_production_v331_patch_v21a.py"
    ).resolve(),
}
MECHANICS_HASHES_V21A = {
    "draw_builder_v21a": mechanics_v21a.DRAW_BUILDER_SHA256_V21A,
    "draw_tests_v21a": mechanics_v21a.DRAW_TEST_SHA256_V21A,
    "draw_plan_v21a": mechanics_v21a.DRAW_FILE_SHA256_V21A,
    "mechanics_v21a": (
        "796f688447e2ac3a1f8483a03343376669fa6ced937e7ee061e85dd8f03cfb91"
    ),
    "mechanics_tests_v21a": (
        "03d19d911b6fdb2f42600c3a5414d742c9985a11f901601866e64c2fbe79852c"
    ),
}

FORBIDDEN_SURFACE_TOKENS_V21A = (
    "checkpoint", "update", "heldout", "validation", "ood", "benchmark",
    "eval", "save-model", "train-dataset", "promote", "union",
)
FORBIDDEN_PERSISTED_KEYS_V21A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "tokens", "row_sha256",
    "ordered_row_identity_sha256", "joint_ids",
    "ordered_joint_identity_sha256", "unit_scores", "responses",
    "coefficients", "bootstrap_replicates", "bootstrap_draws", "row_content",
}
MOE_OVERRIDE_ENVIRONMENT_V21A = driver_v19a.MOE_OVERRIDE_ENVIRONMENT_V19A
RUNTIME_INTEGRITY_KEYS_V21A = {
    "all_four_tp1_engines_every_signed_wave",
    "all_ten_panels_every_direction_sign_and_arm",
    "all_thirty_two_signed_waves_complete",
    "counterbalanced_arm_order_complete",
    "same_resident_perturbation_both_arms",
    "exact_reference_restored_once_per_signed_wave",
    "pre_post_raw_reference_probes_equal",
    "population_boundary_audit_passed",
    "unselected_origin_audit_passed",
    "union_planner_called",
    "all_integrity_audits_passed",
}

canonical_sha256 = prereg_v21a.canonical_sha256
file_sha256 = prereg_v21a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file_v21a(path, commit, digest):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v21a committed runtime input changed: {relative}")


def _verify_commit_bindings_v21a():
    for paths, hashes, commit in (
        (PREREG_PATHS_V21A, PREREG_HASHES_V21A, SEALED_PREREG_COMMIT_V21A),
        (MECHANICS_PATHS_V21A, MECHANICS_HASHES_V21A, SEALED_MECHANICS_COMMIT_V21A),
    ):
        for key, path in paths.items():
            _verify_commit_file_v21a(path, commit, hashes[key])


def implementation_identity_v21a():
    inherited = driver_v19a.implementation_identity_v19a()
    _verify_commit_bindings_v21a()
    paths = {
        **{key: Path(item["path"]) for key, item in inherited["files"].items()},
        **PREREG_PATHS_V21A,
        **MECHANICS_PATHS_V21A,
        "raw_runtime_v21a": Path(__file__).resolve(),
        "raw_runtime_tests_v21a": TEST_PATH_V21A,
    }
    files = {
        key: {"path": str(path), "file_sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    if {
        key: files[key] for key in inherited["files"]
    } != inherited["files"]:
        raise RuntimeError("v21a inherited raw runtime implementation changed")
    return {
        "files": files,
        "inherited_v19a_bundle_sha256": inherited["bundle_sha256"],
        "prereg_bundle_sha256": canonical_sha256({
            key: files[key] for key in PREREG_PATHS_V21A
        }),
        "mechanics_bundle_sha256": canonical_sha256({
            key: files[key] for key in MECHANICS_PATHS_V21A
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def _source_provenance_v21a(implementation):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"v21a real launch requires committed source: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v21a raw source differs from HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    value = {
        "schema": "eggroll-es-raw-source-provenance-v21a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _assert_train_only_argv_v21a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V21A):
            raise ValueError(f"v21a rejects forbidden runtime surface: {token}")


def _declared_moe_backend_v21a():
    return {
        "moe_backend": "default_triton",
        "override_environment": {
            name: None for name in MOE_OVERRIDE_ENVIRONMENT_V21A
        },
        "v16_task_ab_decision_retained": "default_triton",
    }


def _validate_moe_backend_environment_v21a():
    conflicts = {
        name: os.environ.get(name)
        for name in MOE_OVERRIDE_ENVIRONMENT_V21A
        if os.environ.get(name) not in (None, "")
    }
    if conflicts:
        raise ValueError("v21a requires every MoE backend override unset")
    return _declared_moe_backend_v21a()


def resident_signed_wave_schedule_v21a(seeds=None):
    expected = list(prereg_v21a.PERTURBATION_SEEDS_V21A)
    seeds = expected if seeds is None else [int(seed) for seed in seeds]
    if seeds != expected:
        raise RuntimeError("v21a exact fresh 64-direction basis changed")
    schedule = prereg_v21a.signed_wave_schedule_v21a()
    if (
        len(schedule) != 32
        or [item["resident_signed_wave_index"] for item in schedule]
        != list(range(32))
        or sum(len(item["engine_direction_seeds"]) for item in schedule) != 128
    ):
        raise RuntimeError("v21a raw resident schedule changed")
    return schedule


def _load_bound_inputs_v21a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    preregistration = json.loads(prereg_v21a.OUTPUT_PATH_V21A.read_text())
    prereg_v21a.validate_preregistration_v21a(preregistration)
    panel_bundle = mechanics_v21a.load_panel_bundle_v21a()
    frozen = preregistration["frozen_recipe"]
    if (
        Path(frozen["model"]).resolve() != FROZEN_MODEL_V21A
        or frozen["layers"] != [20, 21, 22, 23]
        or frozen["alpha"] != 0.0
        or frozen["sigma"] != 0.0003
        or frozen["population_size"] != 64
        or frozen["layer_plan"]["plan_sha256"] != layer_bundle["plan_sha256"]
        or frozen["layer_plan"]["file_sha256"] != layer_bundle["file_sha256"]
        or frozen["layer_plan"]["model_config_sha256"]
        != layer_bundle["model_config_sha256"]
        or frozen["perturbation_basis_sha256"]
        != prereg_v21a.PERTURBATION_BASIS_SHA256_V21A
        or frozen["perturbation_seed_list_sha256"]
        != prereg_v21a.PERTURBATION_SEED_LIST_SHA256_V21A
        or panel_bundle["content_sha256_before_self_field"]
        != mechanics_v21a.PANEL_BUNDLE_CONTENT_SHA256_V21A
    ):
        raise RuntimeError("v21a model layer basis or panel binding changed")
    return preregistration, panel_bundle


def recipe_v21a(layer, preregistration, panels, implementation, moe):
    if moe != _declared_moe_backend_v21a():
        raise RuntimeError("v21a default Triton declaration changed")
    schedule = resident_signed_wave_schedule_v21a()
    value = {
        "schema": "eggroll-es-authoritative-raw-production-v331-recipe-v21a",
        "experiment_name": EXPERIMENT_NAME_V21A,
        "model": str(FROZEN_MODEL_V21A),
        "preregistration": {
            "commit": SEALED_PREREG_COMMIT_V21A,
            "file_sha256": PREREG_HASHES_V21A["preregistration_v21a"],
            "content_sha256": preregistration["content_sha256_before_self_field"],
        },
        "mechanics_commit": SEALED_MECHANICS_COMMIT_V21A,
        "panel_bundle_content_sha256": panels["content_sha256_before_self_field"],
        "draw_plan_content_sha256": mechanics_v21a.DRAW_CONTENT_SHA256_V21A,
        "layer_plan": copy.deepcopy(preregistration["frozen_recipe"]["layer_plan"]),
        "layers": [20, 21, 22, 23],
        "sigma": 0.0003,
        "alpha": 0.0,
        "population_size": 64,
        "perturbation_basis_sha256": prereg_v21a.PERTURBATION_BASIS_SHA256_V21A,
        "perturbation_seed_list_sha256": (
            prereg_v21a.PERTURBATION_SEED_LIST_SHA256_V21A
        ),
        "signed_wave_schedule_sha256": canonical_sha256(schedule),
        "signed_wave_count": 32,
        "same_perturbation_scores_both_arms_before_restore": True,
        "restore_once_after_both_arms": True,
        "authoritative_raw_scoring": {
            "requests_per_engine_per_signed_wave": 540,
            "requests_by_arm": {
                "production_only": 240,
                "production_plus_v331_patch": 300,
            },
            "requests_per_engine_all_signed_waves": 17_280,
            "dense_result_commitment_count": 2_560,
            "union_planner_called": False,
        },
        "analysis": {
            "contrast_count": 1,
            "endpoint_count": 12,
            "bootstrap_repetitions": 50_000,
            "familywise_one_sided_quantile": 0.05 / 12,
            "noninferiority_margin": 0.0,
        },
        "hardware": {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_every_signed_wave": True,
        },
        "moe_backend": copy.deepcopy(moe),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "authority": {
            "train_only_raw_runtime": True,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
        },
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V21A),
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _parser_v21a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v21a-raw-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_v21a(args, layer, implementation, recipe):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer)
    raw = recipe.get("authoritative_raw_scoring", {})
    if (
        recipe.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(recipe))
        or recipe.get("layers") != [20, 21, 22, 23]
        or recipe.get("alpha") != 0.0
        or recipe.get("sigma") != 0.0003
        or recipe.get("population_size") != 64
        or recipe.get("signed_wave_count") != 32
        or raw.get("requests_per_engine_per_signed_wave") != 540
        or raw.get("requests_per_engine_all_signed_waves") != 17_280
        or raw.get("dense_result_commitment_count") != 2_560
        or raw.get("union_planner_called") is not False
        or any(recipe["authority"][key] is not False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_allowed", "dataset_promotion_allowed",
        ))
    ):
        raise ValueError("v21a raw frozen recipe changed")
    if not args.v21a_raw_dry_run and (
        args.expected_implementation_bundle_sha256 is None
        or args.expected_recipe_sha256 is None
    ):
        raise ValueError("v21a real launch requires exact implementation and recipe hashes")
    if args.expected_implementation_bundle_sha256 not in (
        None, implementation["bundle_sha256"]
    ):
        raise ValueError("v21a raw implementation bundle hash changed")
    if args.expected_recipe_sha256 not in (
        None, recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v21a raw recipe hash changed")


class RawProductionV331AttributionRuntimeMixinV21A(
    driver_v19a.DisjointTierAttributionRuntimeMixinV19A
):
    """Two raw arms under one resident perturbation, with all writes closed."""

    def configure_raw_attribution_v21a(self, panel_bundle, *, frozen_layer_plan):
        panel_bundle = mechanics_v21a.validate_panel_bundle_v21a(
            copy.deepcopy(panel_bundle)
        )
        anchor_v13.validate_frozen_layer_plan_bundle_v13(frozen_layer_plan)
        if (
            len(self.engines) != 4
            or int(self.n_vllm_engines) != 4
            or int(self.n_gpu_per_vllm_engine) != 1
            or int(self.population_size) != 64
            or not math.isclose(float(self.sigma), 0.0003, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, abs_tol=0.0)
        ):
            raise ValueError("v21a requires one four-TP1 alpha-zero trainer")
        reports = self._rpc_all_engines_v4(
            "install_layer_plan_v4",
            (
                Path(frozen_layer_plan["path"]).read_bytes(),
                frozen_layer_plan["file_sha256"],
                frozen_layer_plan["plan_sha256"],
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            ),
        )
        install = anchor_v13.validate_layer_plan_installations_v13(
            reports, frozen_layer_plan
        )
        self._v4_layer_plan = frozen_layer_plan
        self._v4_layer_plan_install = install
        self._v4_reward_config = dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4)
        self._v4_reward_config_sha256 = anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v21a workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4", (4, "exact_reference")
        )
        reference = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(reference, fresh=True)
        self._v21a_panel_bundle = panel_bundle
        return {
            "schema": "eggroll-es-authoritative-raw-runtime-configuration-v21a",
            "layer_plan_install_sha256": canonical_sha256(install),
            "reference_identity_sha256": canonical_sha256(
                reference["reference_identity"]
            ),
            "unselected_origin_sha256": install["unselected_origin_sha256"],
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "union_planner_called": False,
            "train_only_raw_runtime_opened": True,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_allowed": False,
        }

    def _prepared_fixed_batches_v21a(self):
        bundle = mechanics_v21a.validate_panel_bundle_v21a(self._v21a_panel_bundle)
        prepared = {}
        identities = {}
        boundaries = []
        for arm in mechanics_v21a.ARMS_V21A:
            panels = {}
            prompt_items = []
            cursor = 0
            for panel_name in mechanics_v21a.PANEL_NAMES_V21A:
                batch = bundle["panels"][panel_name]["arms"][arm]
                prompts = [base.specialist_template(item) for item in batch["questions"]]
                dense_items = anchor_v4.prepare_gold_answer_items_v4(
                    self.tokenizer, prompts, batch["answers"], max_total_tokens=1024
                )
                for item in dense_items:
                    if (
                        len(item["prompt_token_ids"]) > 1024
                        or item["answer_token_start"] + item["answer_token_count"]
                        != len(item["prompt_token_ids"])
                        or canonical_sha256(item["prompt_token_ids"])
                        != item["prompt_token_ids_sha256"]
                    ):
                        raise RuntimeError("v21a fixed token boundary changed")
                    boundaries.append({
                        "arm": arm, "panel": panel_name,
                        "identity_sha256": item["prompt_token_ids_sha256"],
                        "prompt_token_count": item["prompt_token_count"],
                        "answer_token_start": item["answer_token_start"],
                        "answer_token_count": item["answer_token_count"],
                    })
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(dense_items)),
                }
                prompt_items.extend(dense_items)
                cursor += len(dense_items)
            expected = 10 * frame_v21a.ARM_REQUESTS_PER_PANEL_V21A[arm]
            if cursor != expected:
                raise RuntimeError("v21a fixed raw arm request count changed")
            prepared[arm] = {"panels": panels, "prompt_items": prompt_items}
            identities[arm] = driver_v19a._request_identity_v19a(prompt_items)
        self._v21a_fixed_request_identity = identities
        self._v21a_token_boundary_audit_sha256 = canonical_sha256({
            "schema": "eggroll-es-fixed-token-boundary-audit-v21a",
            "frozen_total_token_cap": 1024,
            "records": boundaries,
        })
        return prepared, identities

    def _assert_fixed_request_v21a(self, arm, prepared):
        if (
            arm not in mechanics_v21a.ARMS_V21A
            or driver_v19a._request_identity_v19a(prepared[arm]["prompt_items"])
            != self._v21a_fixed_request_identity[arm]
        ):
            raise RuntimeError("v21a fixed request identity changed")

    def _raw_reference_probe_v21a(self, prepared):
        commitments = []
        for arm in mechanics_v21a.ARMS_V21A:
            self._assert_fixed_request_v21a(arm, prepared)
            item = prepared[arm]["prompt_items"][0]
            prompt = [{"prompt_token_ids": item["prompt_token_ids"]}]
            batches = self._resolve([
                engine.generate.remote(
                    list(prompt), self._dense_sampling_params_v4(0), use_tqdm=False
                ) for engine in self.engines
            ])
            if len(batches) != 4 or any(len(batch) != 1 for batch in batches):
                raise RuntimeError("v21a raw reference probe incomplete")
            for engine_index, batch in enumerate(batches):
                rewards, dense_hash = driver_v19a._score_panel_unit_outputs_v19a(
                    [item], batch, 1
                )
                commitments.append({
                    "arm": arm, "engine": engine_index, "dense": dense_hash,
                    "score_bytes_sha256": hashlib.sha256(rewards.tobytes()).hexdigest(),
                })
        return canonical_sha256(commitments)

    def _score_resident_arm_v21a(
        self, arm, prepared, schedule_item, unit_scores, dense_commitments,
    ):
        self._assert_fixed_request_v21a(arm, prepared)
        items = prepared[arm]["prompt_items"]
        prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in items]
        batches = self._resolve([
            engine.generate.remote(
                list(prompts), self._dense_sampling_params_v4(0), use_tqdm=False
            ) for engine in self.engines
        ])
        if len(batches) != 4 or any(len(batch) != len(prompts) for batch in batches):
            raise RuntimeError("v21a raw resident arm generation incomplete")
        sign_index = ("plus", "minus").index(schedule_item["sign"])
        for engine_index, outputs in enumerate(batches):
            direction_index = 4 * schedule_item["population_wave_index"] + engine_index
            for panel_index, panel_name in enumerate(mechanics_v21a.PANEL_NAMES_V21A):
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                rewards, dense_hash = driver_v19a._score_panel_unit_outputs_v19a(
                    panel["dense_items"], outputs[start:end], end - start
                )
                unit_scores[arm][panel_index, sign_index, direction_index] = rewards
                dense_commitments.append(dense_hash)
        return canonical_sha256(dense_commitments[-40:])

    def _run_signed_wave_v21a(
        self, schedule_item, prepared, unit_scores, dense_commitments,
    ):
        index = (
            schedule_item.get("resident_signed_wave_index")
            if isinstance(schedule_item, dict) else None
        )
        schedule = resident_signed_wave_schedule_v21a()
        if (
            not isinstance(index, int) or isinstance(index, bool)
            or index not in range(32) or schedule_item != schedule[index]
        ):
            raise RuntimeError("v21a raw signed-wave schedule item changed")
        captures = {}
        try:
            self._perturb_signed_wave_v19a(
                schedule_item["engine_direction_seeds"], schedule_item["negate"]
            )
            for arm in schedule_item["resident_arm_order"]:
                captures[arm] = self._score_resident_arm_v21a(
                    arm, prepared, schedule_item, unit_scores, dense_commitments
                )
        finally:
            restore_hash = self._restore_and_verify_signed_wave_v19a()
        if tuple(captures) != tuple(schedule_item["resident_arm_order"]):
            raise RuntimeError("v21a raw signed-wave transaction changed")
        return restore_hash

    def estimate_raw_attribution_v21a(self, seeds):
        schedule = resident_signed_wave_schedule_v21a(seeds)
        prepared, request_identity = self._prepared_fixed_batches_v21a()
        pre_probe = self._raw_reference_probe_v21a(prepared)
        unit_scores = {
            arm: np.full(
                (10, 2, 64, frame_v21a.ARM_REQUESTS_PER_PANEL_V21A[arm]),
                np.nan, dtype=np.float64,
            ) for arm in mechanics_v21a.ARMS_V21A
        }
        dense_commitments = []
        restore_hashes = []
        for item in schedule:
            restore_hashes.append(self._run_signed_wave_v21a(
                item, prepared, unit_scores, dense_commitments
            ))
        if (
            any(not np.isfinite(values).all() for values in unit_scores.values())
            or len(dense_commitments) != 2_560
            or len(restore_hashes) != 32
        ):
            raise RuntimeError("v21a raw population capture incomplete")
        boundary = self._population_boundary_audit_v4(0)
        unselected, unselected_sha = driver_v19a._validate_population_boundary_v19a(
            boundary, self._v4_layer_plan_install
        )
        post_probe = self._raw_reference_probe_v21a(prepared)
        if pre_probe != post_probe:
            raise RuntimeError("v21a raw pre/post reference probe drifted")
        compact = mechanics_v21a.build_compact_estimator_summary_v21a(
            unit_scores, self._v21a_panel_bundle
        )
        unit_scores = None
        integrity = {
            "all_four_tp1_engines_every_signed_wave": True,
            "all_ten_panels_every_direction_sign_and_arm": True,
            "all_thirty_two_signed_waves_complete": True,
            "counterbalanced_arm_order_complete": True,
            "same_resident_perturbation_both_arms": True,
            "exact_reference_restored_once_per_signed_wave": True,
            "pre_post_raw_reference_probes_equal": True,
            "population_boundary_audit_passed": True,
            "unselected_origin_audit_passed": unselected["passed"],
            "union_planner_called": False,
            "all_integrity_audits_passed": True,
        }
        summary = {
            "schema": "eggroll-es-raw-production-v331-attribution-summary-v21a",
            "experiment_name": EXPERIMENT_NAME_V21A,
            "alpha": 0.0, "sigma": 0.0003,
            "runtime_integrity": integrity,
            "arms": compact["arms"],
            "paired_bootstrap": compact["paired_bootstrap"],
            "union_planner_called": False,
            "model_update_applied": False, "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_applied": False,
            "persisted_response_vectors_or_row_content": False,
            "bootstrap_draws_persisted": False,
            "unit_scores_persisted": False,
        }
        summary["content_sha256_before_self_field"] = canonical_sha256(summary)
        gate = mechanics_v21a.evaluate_compatibility_gate_v21a(summary)
        audit = {
            "schema": "eggroll-es-raw-production-v331-runtime-audit-v21a",
            "fixed_request_identity_sha256": canonical_sha256(request_identity),
            "token_boundary_audit_sha256": self._v21a_token_boundary_audit_sha256,
            "pre_post_probe_identity_sha256": canonical_sha256({
                "pre": pre_probe, "post": post_probe
            }),
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "dense_result_commitments_sha256": canonical_sha256(dense_commitments),
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "unselected_origin_sha256": boundary["unselected_origin_sha256"],
            "unselected_origin_audit_sha256": unselected_sha,
            "signed_wave_count": 32, "panel_count": 10,
            "requests_per_engine_per_signed_wave": 540,
            "requests_per_engine_all_signed_waves": 17_280,
            "dense_result_commitment_count": 2_560,
            "union_planner_called": False,
            "per_unit_scores_persisted": False,
            "bootstrap_replicates_persisted": False,
            "bootstrap_draws_persisted": False,
            "row_content_persisted": False,
        }
        audit["content_sha256_before_self_field"] = canonical_sha256(audit)
        return summary, gate, audit

    @staticmethod
    def _closed_surface_v21a(*_args, **_kwargs):
        raise RuntimeError(
            "v21a closes union update checkpoint evaluation and promotion surfaces"
        )

    configure_disjoint_tier_attribution_v19a = _closed_surface_v21a
    configure_production_patch_compat_v18a = _closed_surface_v21a
    run_union_equivalence_v20a = _closed_surface_v21a
    configure_union_equivalence_v20a = _closed_surface_v21a
    build_union_request_plan_v20a = _closed_surface_v21a
    _generate_union_v20a = _closed_surface_v21a
    estimate_disjoint_tier_attribution_v19a = _closed_surface_v21a
    estimate_production_patch_compat_v18a = _closed_surface_v21a
    estimate_train_panels_v13 = _closed_surface_v21a
    estimate_step_coefficients = _closed_surface_v21a
    apply_seed_coefficients = _closed_surface_v21a
    _evaluate_population_with_anchor = _closed_surface_v21a
    _persist_anchor_plan = _closed_surface_v21a
    _persist_identity_audit = _closed_surface_v21a
    _abort_update_v3 = _closed_surface_v21a
    _abort_update_v4 = _closed_surface_v21a
    train_step = _closed_surface_v21a
    evaluate_handle = _closed_surface_v21a
    evaluate_population_on_batch = _closed_surface_v21a
    eval_step = _closed_surface_v21a
    fit = _closed_surface_v21a


def load_runtime_trainer_v21a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    parent = anchor_v13.load_trainer(layer_bundle)

    class RawProductionV331AttributionRuntimeTrainerV21A(
        RawProductionV331AttributionRuntimeMixinV21A, parent
    ):
        pass

    return RawProductionV331AttributionRuntimeTrainerV21A


def _make_trainer_v21a(layer_bundle):
    trainer_class = load_runtime_trainer_v21a(layer_bundle)
    return trainer_class(
        model_name=str(FROZEN_MODEL_V21A), checkpoint=None, sigma=0.0003,
        alpha=0.0, population_size=64, reward_shaping="z-scores",
        num_iterations=1, max_tokens=1, batch_size=300, mini_batch_size=300,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template, train_dataloader=[],
        eval_dataloader_dict={}, eval_freq=1, n_vllm_engines=4,
        n_gpu_per_vllm_engine=1, logging="none", global_seed=43,
        use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_V21A,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(FROZEN_OUTPUT_DIRECTORY_V21A),
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def validate_compact_report_v21a(report, *, expected_recipe, expected_implementation):
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    audit = report.get("runtime_audit", {}) if isinstance(report, dict) else {}
    config = report.get("configuration", {}) if isinstance(report, dict) else {}
    integrity = summary.get("runtime_integrity", {})
    expected_keys = {
        "schema", "recipe", "configuration", "runtime_audit", "summary",
        "gate", "implementation", "union_planner_called",
        "model_update_applied", "checkpoint_written",
        "evaluation_surfaces_opened", "dataset_promotion_applied",
        "content_sha256_before_self_field",
    }
    if (
        not isinstance(report, dict) or set(report) != expected_keys
        or report.get("schema")
        != "eggroll-es-authoritative-raw-production-v331-report-v21a"
        or report.get("recipe") != expected_recipe
        or report.get("implementation") != expected_implementation
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(report))
        or summary.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(summary))
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(audit))
        or report.get("gate")
        != mechanics_v21a.evaluate_compatibility_gate_v21a(summary)
        or set(integrity) != RUNTIME_INTEGRITY_KEYS_V21A
        or integrity.get("union_planner_called") is not False
        or any(
            integrity.get(key) is not True
            for key in RUNTIME_INTEGRITY_KEYS_V21A - {"union_planner_called"}
        )
        or any(value.get("union_planner_called") is not False for value in (
            report, summary, audit, config
        ))
        or audit.get("signed_wave_count") != 32
        or audit.get("requests_per_engine_per_signed_wave") != 540
        or audit.get("requests_per_engine_all_signed_waves") != 17_280
        or audit.get("dense_result_commitment_count") != 2_560
        or config.get("engine_count") != 4 or config.get("tp_per_engine") != 1
        or config.get("gpu_ids") != [0, 1, 2, 3]
        or config.get("train_only_raw_runtime_opened") is not True
        or any(config.get(key) is not False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_surfaces_opened", "dataset_promotion_allowed",
        ))
        or any(report.get(key) is not False for key in (
            "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
        ))
        or any(
            str(key).lower() in FORBIDDEN_PERSISTED_KEYS_V21A
            for key in _recursive_keys(report)
        )
    ):
        raise RuntimeError("v21a raw compact report changed")
    return report


def _seal(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)


def _exclusive_write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal(value)
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json(path, value):
    _seal(value)
    driver_v19a.driver_v13.driver_v1.atomic_write_json(path, value)


def _claim_paths(attempt):
    root = (FROZEN_OUTPUT_DIRECTORY_V21A / EXPERIMENT_NAME_V21A).resolve()
    for _ in range(32):
        launch = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"
        attempt_path = root / "attempts" / f"{launch}.json"
        run_dir = root / "runs" / launch
        current = copy.deepcopy(attempt)
        current.update({"launch_id": launch, "run_directory": str(run_dir)})
        try:
            _exclusive_write_json(attempt_path, current)
        except FileExistsError:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        return attempt_path, run_dir, current
    raise RuntimeError("v21a raw could not claim fresh paths")


def _record_failure(attempt_path, attempt, report_path, failure, phase):
    attempt.update({
        "status": "failed", "phase": phase,
        "failure_type": type(failure).__name__,
        "failure_sha256": canonical_sha256({
            "type": type(failure).__name__, "repr": repr(failure)
        }),
        "report_exists_after_attempt": report_path.exists(),
    })
    _rewrite_json(attempt_path, attempt)


def run_exact_v21a(layer, panels, implementation, recipe):
    attempt = {
        "schema": "eggroll-es-raw-attribution-attempt-v21a",
        "status": "launching", "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V21A, "recipe": recipe,
        "source_provenance": _source_provenance_v21a(implementation),
        "union_planner_called": False,
        "model_update_applied": False, "checkpoint_written": False,
        "evaluation_surfaces_opened": False, "dataset_promotion_applied": False,
    }
    attempt_path, run_dir, attempt = _claim_paths(attempt)
    report_path = run_dir / REPORT_NAME_V21A
    trainer = None
    failure = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v21a(layer)
        configuration = trainer.configure_raw_attribution_v21a(
            panels, frozen_layer_plan=layer
        )
        summary, gate, audit = trainer.estimate_raw_attribution_v21a(
            prereg_v21a.PERTURBATION_SEEDS_V21A
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
        _record_failure(
            attempt_path, attempt, report_path, failure,
            "inside_raw_runtime_or_cleanup",
        )
        raise failure
    try:
        report = {
            "schema": "eggroll-es-authoritative-raw-production-v331-report-v21a",
            "recipe": recipe, "configuration": configuration,
            "runtime_audit": audit, "summary": summary, "gate": gate,
            "implementation": implementation, "union_planner_called": False,
            "model_update_applied": False, "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_applied": False,
        }
        report["content_sha256_before_self_field"] = canonical_sha256(report)
        validate_compact_report_v21a(
            report, expected_recipe=recipe, expected_implementation=implementation
        )
        _exclusive_write_json(report_path, report)
    except BaseException as error:
        _record_failure(
            attempt_path, attempt, report_path, error,
            "after_cleanup_before_compact_report_completion",
        )
        raise
    attempt.update({
        "status": "complete", "phase": "after_cleanup_and_compact_report",
        "report_exists_after_attempt": True,
        "report_binding": {
            "path": str(report_path), "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
    })
    _rewrite_json(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v21a(argv)
    moe = _validate_moe_backend_environment_v21a()
    layer, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v21a().parse_args(remaining)
    implementation = implementation_identity_v21a()
    preregistration, panels = _load_bound_inputs_v21a(layer)
    recipe = recipe_v21a(layer, preregistration, panels, implementation, moe)
    validate_runtime_v21a(args, layer, implementation, recipe)
    if args.v21a_raw_dry_run:
        payload = {
            "schema": "eggroll-es-authoritative-raw-attribution-dry-run-v21a",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "implementation": implementation, "recipe": recipe,
            "gpu_launched": False, "union_planner_called": False,
        }
        payload["content_sha256_before_self_field"] = canonical_sha256(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v21a(layer, panels, implementation, recipe)


if __name__ == "__main__":
    main()
