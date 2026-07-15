#!/usr/bin/env python3
"""Fail-closed train-only runtime for V19A disjoint-tier attribution."""

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

import run_eggroll_es_production_patch_compat_v18a as driver_v18a
import train_eggroll_es_disjoint_tier_attribution_v19a as mechanics_v19a


ROOT = Path(__file__).resolve().parent
prereg_v19a = mechanics_v19a.prereg_v19a
frame_v19a = mechanics_v19a.frame_v19a
driver_v13 = driver_v18a.driver_v13
anchor_v13 = driver_v18a.anchor_v13
anchor_v11 = driver_v18a.anchor_v11
anchor_v4 = driver_v18a.anchor_v4
base = driver_v18a.base

FROZEN_MODEL_V19A = driver_v18a.FROZEN_MODEL_V18A
FROZEN_OUTPUT_DIRECTORY_V19A = driver_v18a.FROZEN_OUTPUT_DIRECTORY_V18A
EXPERIMENT_NAME_V19A = prereg_v19a.EXPERIMENT_NAME_V19A
REPORT_NAME_V19A = "disjoint_tier_attribution_v19a.json"
TEST_PATH_V19A = (
    ROOT / "test_run_eggroll_es_disjoint_tier_attribution_v19a.py"
).resolve()
TRAINER_PATH_V19A = Path(mechanics_v19a.__file__).resolve()
TRAINER_TEST_PATH_V19A = (
    ROOT / "test_train_eggroll_es_disjoint_tier_attribution_v19a.py"
).resolve()

SEALED_FRAME_PREREG_COMMIT_V19A = (
    "97cddf65df07ce1c87e3838272f2558bcb53d911"
)
SEALED_TRAINER_COMMIT_V19A = "28e9a424c526db989560b76ba0504f143a0a17d6"
OFFLINE_PHASE_HASHES_V19A = {
    "frame_builder_v19a": mechanics_v19a.FRAME_BUILDER_FILE_SHA256_V19A,
    "frame_tests_v19a": (
        "2034c1d1e214589a057197ffe1cfb0d19d803cb14b6d8b1418bee4cb034466d0"
    ),
    "frame_certificate_v19a": mechanics_v19a.FRAME_CERTIFICATE_FILE_SHA256_V19A,
    "prereg_module_v19a": mechanics_v19a.PREREGISTRATION_BUILDER_FILE_SHA256_V19A,
    "prereg_tests_v19a": (
        "3b746d5798942d493eedea0a7f887ec09ae29f5d5e633afe968453abf555d4d5"
    ),
    "preregistration_v19a": mechanics_v19a.PREREGISTRATION_FILE_SHA256_V19A,
}
OFFLINE_PHASE_PATHS_V19A = {
    "frame_builder_v19a": Path(frame_v19a.__file__).resolve(),
    "frame_tests_v19a": (
        ROOT / "test_build_eggroll_es_disjoint_tier_frame_v19a.py"
    ).resolve(),
    "frame_certificate_v19a": frame_v19a.OUTPUT_PATH_V19A,
    "prereg_module_v19a": Path(prereg_v19a.__file__).resolve(),
    "prereg_tests_v19a": (
        ROOT / "test_eggroll_es_disjoint_tier_attribution_preregistration_v19a.py"
    ).resolve(),
    "preregistration_v19a": prereg_v19a.OUTPUT_PATH_V19A,
}
TRAINER_PHASE_HASHES_V19A = {
    "trainer_mechanics_v19a": (
        "38db195e18f8b2dd9483c77a7e93d8026d8fb6da2319b158b38a6e2218b17924"
    ),
    "trainer_tests_v19a": (
        "1e9493a84ef4f55d4390eb00e868b48897d2a573508767baab1043779ba5c25c"
    ),
}
TRAINER_PHASE_PATHS_V19A = {
    "trainer_mechanics_v19a": TRAINER_PATH_V19A,
    "trainer_tests_v19a": TRAINER_TEST_PATH_V19A,
}
IMPLEMENTATION_PATHS_V19A = {
    **driver_v18a.IMPLEMENTATION_PATHS_V18A,
    **OFFLINE_PHASE_PATHS_V19A,
    **TRAINER_PHASE_PATHS_V19A,
    "runtime_driver_v19a": Path(__file__).resolve(),
    "runtime_tests_v19a": TEST_PATH_V19A,
}

FORBIDDEN_SURFACE_TOKENS_V19A = (
    "checkpoint",
    "update",
    "heldout",
    "validation",
    "ood",
    "benchmark",
    "eval",
    "save-model",
    "train-dataset",
    "promote",
)
FORBIDDEN_PERSISTED_KEYS_V19A = {
    "question",
    "questions",
    "answer",
    "answers",
    "prompt",
    "prompts",
    "prompt_token_ids",
    "token_ids",
    "tokens",
    "row_sha256",
    "ordered_row_identity_sha256",
    "joint_ids",
    "ordered_joint_identity_sha256",
    "unit_scores",
    "responses",
    "coefficients",
    "bootstrap_replicates",
    "bootstrap_draws",
    "row_content",
}
MOE_OVERRIDE_ENVIRONMENT_V19A = driver_v18a.MOE_OVERRIDE_ENVIRONMENT_V18A

canonical_sha256 = prereg_v19a.canonical_sha256
file_sha256 = prereg_v19a.file_sha256


def _without_self_v19a(value):
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _assert_train_only_argv_v19a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V19A):
            raise ValueError(f"v19a rejects forbidden runtime surface: {token}")


def _verify_commit_file_v19a(path: Path, commit: str, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v19a committed artifact changed: {relative}")


def _verify_commit_bindings_v19a():
    for key, path in OFFLINE_PHASE_PATHS_V19A.items():
        _verify_commit_file_v19a(
            path,
            SEALED_FRAME_PREREG_COMMIT_V19A,
            OFFLINE_PHASE_HASHES_V19A[key],
        )
    for key, path in TRAINER_PHASE_PATHS_V19A.items():
        _verify_commit_file_v19a(
            path,
            SEALED_TRAINER_COMMIT_V19A,
            TRAINER_PHASE_HASHES_V19A[key],
        )


def implementation_identity_v19a():
    inherited = driver_v18a.implementation_identity_v18a()
    _verify_commit_bindings_v19a()
    mechanics_v19a.load_hardened_preregistration_v19a()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V19A.items()
    }
    if {key: files[key] for key in inherited["files"]} != inherited["files"]:
        raise RuntimeError("v19a inherited V18A implementation changed")
    expected = {**OFFLINE_PHASE_HASHES_V19A, **TRAINER_PHASE_HASHES_V19A}
    if {key: files[key]["file_sha256"] for key in expected} != expected:
        raise RuntimeError("v19a sealed offline or trainer phase changed")
    offline_phase = {key: files[key] for key in OFFLINE_PHASE_PATHS_V19A}
    trainer_phase = {key: files[key] for key in TRAINER_PHASE_PATHS_V19A}
    return {
        "files": files,
        "inherited_v18a_bundle_sha256": inherited["bundle_sha256"],
        "offline_phase_bundle_sha256": canonical_sha256(offline_phase),
        "trainer_phase_bundle_sha256": canonical_sha256(trainer_phase),
        "bundle_sha256": canonical_sha256(files),
    }


def _source_provenance_v19a(implementation):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"v19a real launch requires committed source: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v19a source differs from HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    result = {
        "schema": "eggroll-es-committed-source-bundle-v19a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _parser_v19a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def _declared_moe_backend_v19a():
    return {
        "moe_backend": "default_triton",
        "override_environment": {
            name: None for name in MOE_OVERRIDE_ENVIRONMENT_V19A
        },
        "v16_task_ab_decision_retained": "default_triton",
    }


def _validate_moe_backend_environment_v19a():
    conflicts = {
        name: os.environ.get(name)
        for name in MOE_OVERRIDE_ENVIRONMENT_V19A
        if os.environ.get(name) not in (None, "")
    }
    if conflicts:
        raise ValueError("v19a requires every MoE backend override unset")
    return _declared_moe_backend_v19a()


def resident_signed_wave_schedule_v19a(seeds=None) -> list[dict]:
    seeds = list(prereg_v19a.PERTURBATION_SEEDS_V19A if seeds is None else seeds)
    if seeds != prereg_v19a.PERTURBATION_SEEDS_V19A:
        raise RuntimeError("v19a exact fresh perturbation basis changed")
    orders = prereg_v19a.signed_wave_arm_orders_v19a()
    schedule = []
    for wave_index, start in enumerate(range(0, len(seeds), 4)):
        wave = [int(seed) for seed in seeds[start:start + 4]]
        if len(wave) != 4:
            raise RuntimeError("v19a partial four-engine wave is forbidden")
        for sign_index, sign in enumerate(("plus", "minus")):
            signed_wave_index = 2 * wave_index + sign_index
            schedule.append({
                "signed_wave_index": signed_wave_index,
                "population_wave_index": wave_index,
                "sign": sign,
                "negate": sign == "minus",
                "engine_seeds": wave,
                "resident_arm_order": orders[signed_wave_index]["arm_order"],
                "restore_after_all_four_arms": True,
            })
    if (
        len(schedule) != 16
        or [item["signed_wave_index"] for item in schedule] != list(range(16))
        or any(
            set(item["resident_arm_order"]) != set(mechanics_v19a.ARMS_V19A)
            for item in schedule
        )
    ):
        raise RuntimeError("v19a resident four-arm schedule changed")
    return schedule


def _load_bound_inputs_v19a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    plan = anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    preregistration = mechanics_v19a.load_hardened_preregistration_v19a()
    panel_bundle = mechanics_v19a.load_panel_bundle_v19a()
    frozen = preregistration["frozen_recipe"]
    if (
        layer_bundle["plan_sha256"]
        != driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
        or Path(layer_bundle["path"]).resolve() != Path(plan["path"]).resolve()
        or layer_bundle["file_sha256"] != plan["file_sha256"]
        or layer_bundle["model_config_sha256"]
        != prereg_v19a.prereg_v18a.prereg_v17a.MODEL_CONFIG_SHA256_V17A
        or Path(frozen["model"]).resolve() != FROZEN_MODEL_V19A
        or frozen["layers"] != [20, 21, 22, 23]
        or frozen["alpha"] != 0.0
        or frozen["sigma"] != 0.0003
        or frozen["population_size"] != 32
        or frozen["layer_plan"]["plan_sha256"] != layer_bundle["plan_sha256"]
        or frozen["perturbation_basis_sha256"]
        != prereg_v19a.PERTURBATION_BASIS_SHA256_V19A
        or frozen["perturbation_seed_list_sha256"]
        != prereg_v19a.PERTURBATION_SEED_LIST_SHA256_V19A
        or frozen["perturbation_basis"]["seeds"]
        != prereg_v19a.PERTURBATION_SEEDS_V19A
        or frozen["prior_v18a_basis_reuse_allowed"] is not False
        or preregistration["scoring"]["dense_reward_config_sha256"]
        != anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        or panel_bundle["content_sha256_before_self_field"]
        != mechanics_v19a.PANEL_BUNDLE_CONTENT_SHA256_V19A
        or panel_bundle["token_audit"]["observed_combined_token_max"] > 1024
    ):
        raise RuntimeError("v19a model/layer/basis/scoring binding changed")
    return preregistration, panel_bundle


def recipe_v19a(
    layer_bundle, preregistration, panel_bundle, implementation, moe_backend,
):
    if moe_backend != _declared_moe_backend_v19a():
        raise RuntimeError("v19a default Triton backend declaration changed")
    panels = {
        panel_name: {
            "role": panel_bundle["panels"][panel_name]["role"],
            "arm_rows": {
                arm: len(panel_bundle["panels"][panel_name]["arms"][arm]["questions"])
                for arm in mechanics_v19a.ARMS_V19A
            },
        }
        for panel_name in mechanics_v19a.PANEL_NAMES_V19A
    }
    schedule = resident_signed_wave_schedule_v19a()
    recipe = {
        "schema": "eggroll-es-disjoint-tier-attribution-recipe-v19a",
        "experiment_name": EXPERIMENT_NAME_V19A,
        "model": str(FROZEN_MODEL_V19A),
        "preregistration": copy.deepcopy(panel_bundle["preregistration"]),
        "frame": copy.deepcopy(panel_bundle["frame"]),
        "sources": copy.deepcopy(panel_bundle["sources"]),
        "token_audit": copy.deepcopy(panel_bundle["token_audit"]),
        "materialized_panel_bundle_content_sha256": panel_bundle[
            "content_sha256_before_self_field"
        ],
        "panels": panels,
        "layer_plan": copy.deepcopy(preregistration["frozen_recipe"]["layer_plan"]),
        "perturbation": {
            "layers": [20, 21, 22, 23],
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 32,
            "basis_seed": prereg_v19a.FRESH_PERTURBATION_BASIS_SEED_V19A,
            "basis_sha256": prereg_v19a.PERTURBATION_BASIS_SHA256_V19A,
            "seed_list_sha256": prereg_v19a.PERTURBATION_SEED_LIST_SHA256_V19A,
            "seeds": list(prereg_v19a.PERTURBATION_SEEDS_V19A),
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "score_all_four_arms_before_restore": True,
            "restore_once_after_all_four_arms": True,
            "basis_generation_or_selection_at_launch_allowed": False,
        },
        "scoring": {
            "objective": "per_example_mean_full_gold_answer_token_logprob",
            "dense_reward_config_sha256": anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
            "frozen_total_prompt_answer_token_cap": 1024,
            "requests_per_engine_per_arm_per_signed_wave": {
                "production_only": 240,
                "patch_tier_1_only": 250,
                "patch_tier_2_only": 250,
                "patch_tier_3_only": 250,
            },
            "requests_per_engine_per_signed_wave_all_arms": 990,
            "dense_result_commitment_count": 2560,
        },
        "bootstrap": copy.deepcopy(preregistration["analysis"]["bootstrap"]),
        "bootstrap_draw_plan_content_sha256": (
            mechanics_v19a.BOOTSTRAP_DRAW_PLAN_CONTENT_SHA256_V19A
        ),
        "hardware": {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        },
        "moe_backend": copy.deepcopy(moe_backend),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "trainer_phase_bundle_sha256": implementation[
            "trainer_phase_bundle_sha256"
        ],
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_allowed": False,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V19A),
    }
    recipe["content_sha256_before_self_field"] = canonical_sha256(recipe)
    prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(
        recipe
    )
    return recipe


def validate_runtime_v19a(args, layer_bundle, implementation, recipe):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    if (
        not isinstance(args.v19a_dry_run, bool)
        or recipe.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v19a(recipe))
        or recipe.get("model_update_allowed") is not False
        or recipe.get("checkpoint_write_allowed") is not False
        or recipe.get("evaluation_surfaces_opened") is not False
        or recipe.get("dataset_promotion_allowed") is not False
        or recipe.get("moe_backend") != _declared_moe_backend_v19a()
        or recipe.get("hardware")
        != {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        }
        or recipe.get("scoring", {}).get(
            "requests_per_engine_per_signed_wave_all_arms"
        )
        != 990
        or recipe.get("scoring", {}).get("dense_result_commitment_count")
        != 2560
        or recipe.get("perturbation", {}).get("seeds")
        != prereg_v19a.PERTURBATION_SEEDS_V19A
        or recipe.get("perturbation", {}).get("basis_sha256")
        != prereg_v19a.PERTURBATION_BASIS_SHA256_V19A
        or recipe.get("bootstrap_draw_plan_content_sha256")
        != mechanics_v19a.BOOTSTRAP_DRAW_PLAN_CONTENT_SHA256_V19A
    ):
        raise ValueError("v19a frozen train-only runtime recipe changed")
    if not args.v19a_dry_run and (
        args.expected_implementation_bundle_sha256 is None
        or args.expected_recipe_sha256 is None
    ):
        raise ValueError("v19a real launch requires implementation and recipe hashes")
    if (
        args.expected_implementation_bundle_sha256 is not None
        and args.expected_implementation_bundle_sha256
        != implementation["bundle_sha256"]
    ):
        raise ValueError("v19a implementation bundle hash changed")
    if (
        args.expected_recipe_sha256 is not None
        and args.expected_recipe_sha256 != recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v19a recipe hash changed")


def _request_identity_v19a(prompt_items):
    return canonical_sha256([
        item["prompt_token_ids_sha256"] for item in prompt_items
    ])


def _score_panel_unit_outputs_v19a(dense_items, outputs, expected_rows):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = np.asarray([
        item["mean_answer_token_logprob"] for item in dense["examples"]
    ], dtype=np.float64)
    if rewards.shape != (expected_rows,) or not np.isfinite(rewards).all():
        raise RuntimeError("v19a per-unit dense score coverage changed")
    return rewards, canonical_sha256(dense)


def _validate_population_boundary_v19a(boundary, layer_install):
    expected_origin = layer_install.get("unselected_origin_sha256")
    reports = boundary.get("worker_reports", []) if isinstance(boundary, dict) else []
    if (
        not isinstance(boundary, dict)
        or boundary.get("passed") is not True
        or boundary.get("engine_count") != 4
        or boundary.get("audit_sha256")
        != canonical_sha256({
            key: value for key, value in boundary.items() if key != "audit_sha256"
        })
        or not isinstance(expected_origin, str)
        or boundary.get("unselected_origin_sha256") != expected_origin
        or boundary.get("runtime_mapping") != layer_install
        or len(reports) != 4
        or sorted(report.get("rank") for report in reports) != list(range(4))
        or any(
            report.get("passed") is not True
            or any(report.get(key) != value for key, value in layer_install.items())
            for report in reports
        )
    ):
        raise RuntimeError("v19a population or unselected-origin audit failed")
    payload = {
        "schema": "eggroll-es-unselected-origin-audit-v19a",
        "population_boundary_audit_sha256": boundary["audit_sha256"],
        "unselected_origin_sha256": expected_origin,
        "worker_rank_origin_sha256": canonical_sha256([
            {
                "rank": report.get("rank"),
                "unselected_origin_sha256": report.get("unselected_origin_sha256"),
            }
            for report in sorted(reports, key=lambda item: item.get("rank"))
        ]),
        "passed": True,
    }
    return payload, canonical_sha256(payload)


class DisjointTierAttributionRuntimeMixinV19A(
    driver_v18a.ProductionPatchCompatRuntimeMixinV18A
):
    """Resident four-arm scorer with every mutation/evaluation surface closed."""

    def configure_disjoint_tier_attribution_v19a(
        self, panel_bundle, *, frozen_layer_plan,
    ):
        panel_bundle = mechanics_v19a.validate_panel_bundle_v19a(
            copy.deepcopy(panel_bundle)
        )
        anchor_v13.validate_frozen_layer_plan_bundle_v13(frozen_layer_plan)
        if (
            len(self.engines) != 4
            or int(self.n_vllm_engines) != 4
            or int(self.n_gpu_per_vllm_engine) != 1
            or int(self.population_size) != 32
            or not math.isclose(float(self.sigma), 0.0003, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, abs_tol=0.0)
        ):
            raise ValueError("v19a requires one four-TP1-engine alpha-zero trainer")
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
            raise RuntimeError("v19a workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4", (4, "exact_reference")
        )
        reference = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(reference, fresh=True)
        self._v19a_panel_bundle = panel_bundle
        return {
            "schema": "eggroll-es-disjoint-tier-runtime-configuration-v19a",
            "layer_plan_install_sha256": canonical_sha256(install),
            "reference_identity_sha256": canonical_sha256(
                reference["reference_identity"]
            ),
            "unselected_origin_sha256": install["unselected_origin_sha256"],
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "worker_extension": (
                "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
            ),
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_allowed": False,
        }

    def _prepared_fixed_batches_v19a(self):
        bundle = mechanics_v19a.validate_panel_bundle_v19a(
            self._v19a_panel_bundle
        )
        prepared = {}
        identities = {}
        boundary_records = []
        for arm in mechanics_v19a.ARMS_V19A:
            panels = {}
            prompt_items = []
            cursor = 0
            for panel_name in mechanics_v19a.PANEL_NAMES_V19A:
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
                        raise RuntimeError("v19a fixed token boundary changed")
                    boundary_records.append({
                        "arm": arm,
                        "panel": panel_name,
                        "prompt_token_ids_sha256": item["prompt_token_ids_sha256"],
                        "prompt_token_count": item["prompt_token_count"],
                        "answer_token_start": item["answer_token_start"],
                        "answer_token_count": item["answer_token_count"],
                    })
                current = [{"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items]
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(current)),
                }
                prompt_items.extend(dense_items)
                cursor += len(current)
            expected = 10 * frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm]
            if cursor != expected:
                raise RuntimeError("v19a fixed arm request count changed")
            prepared[arm] = {"panels": panels, "prompt_items": prompt_items}
            identities[arm] = _request_identity_v19a(prompt_items)
        self._v19a_fixed_request_identity = identities
        self._v19a_token_boundary_audit_sha256 = canonical_sha256({
            "schema": "eggroll-es-fixed-token-boundary-audit-v19a",
            "frozen_total_token_cap": 1024,
            "records": boundary_records,
        })
        return prepared, identities

    def _assert_fixed_request_v19a(self, arm, prepared):
        if (
            arm not in mechanics_v19a.ARMS_V19A
            or _request_identity_v19a(prepared[arm]["prompt_items"])
            != self._v19a_fixed_request_identity[arm]
        ):
            raise RuntimeError("v19a fixed request identity changed")

    def _base_probe_v19a(self, prepared):
        result = {}
        for arm in mechanics_v19a.ARMS_V19A:
            self._assert_fixed_request_v19a(arm, prepared)
            prompt_items = [
                {"prompt_token_ids": item["prompt_token_ids"]}
                for item in prepared[arm]["prompt_items"]
            ]
            outputs = anchor_v13.anchor_v11.anchor_v1.dispatch_eval_batch(
                self.engines,
                prompt_items,
                self._dense_sampling_params_v4(0),
                self._resolve,
            )
            panel_hashes = {}
            for panel_name in mechanics_v19a.PANEL_NAMES_V19A:
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                _rewards, dense_hash = _score_panel_unit_outputs_v19a(
                    panel["dense_items"], outputs[start:end], end - start
                )
                panel_hashes[panel_name] = dense_hash
            result[arm] = canonical_sha256(panel_hashes)
        return result

    def _perturb_signed_wave_v19a(self, engine_seeds, negate):
        if len(engine_seeds) != 4:
            raise RuntimeError("v19a perturbation wave is partial")
        self._resolve([
            self.engines[index].collective_rpc.remote(
                "perturb_self_weights", args=(int(seed), 0.0003, bool(negate))
            )
            for index, seed in enumerate(engine_seeds)
        ])

    def _score_resident_arm_v19a(
        self, arm, prepared, schedule_item, unit_scores, dense_commitments,
    ):
        self._assert_fixed_request_v19a(arm, prepared)
        prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in prepared[arm]["prompt_items"]
        ]
        batches = self._resolve([
            engine.generate.remote(
                list(prompts), self._dense_sampling_params_v4(0), use_tqdm=False
            )
            for engine in self.engines
        ])
        if (
            not isinstance(batches, list)
            or len(batches) != 4
            or any(len(batch) != len(prompts) for batch in batches)
        ):
            raise RuntimeError("v19a resident arm generation is incomplete")
        sign_index = ("plus", "minus").index(schedule_item["sign"])
        for engine_index, batch in enumerate(batches):
            direction_index = 4 * schedule_item["population_wave_index"] + engine_index
            for panel_index, panel_name in enumerate(mechanics_v19a.PANEL_NAMES_V19A):
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                rewards, dense_hash = _score_panel_unit_outputs_v19a(
                    panel["dense_items"], batch[start:end], end - start
                )
                unit_scores[arm][panel_index, sign_index, direction_index] = rewards
                dense_commitments.append(dense_hash)
        return canonical_sha256(dense_commitments[-40:])

    def _restore_and_verify_signed_wave_v19a(self):
        self._restore_all_engines_exact()
        checks = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        if not anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks,
            lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v19a signed-wave exact reference restore failed")
        return canonical_sha256(checks)

    def _run_signed_wave_v19a(
        self, schedule_item, prepared, unit_scores, dense_commitments,
    ):
        signed_wave_index = (
            schedule_item.get("signed_wave_index")
            if isinstance(schedule_item, dict)
            else None
        )
        schedule = resident_signed_wave_schedule_v19a()
        if (
            not isinstance(signed_wave_index, int)
            or isinstance(signed_wave_index, bool)
            or signed_wave_index not in range(len(schedule))
            or schedule_item != schedule[signed_wave_index]
        ):
            raise RuntimeError("v19a signed-wave schedule item changed")
        captures = {}
        restore_hash = None
        try:
            self._perturb_signed_wave_v19a(
                list(schedule_item["engine_seeds"]), bool(schedule_item["negate"])
            )
            for arm in schedule_item["resident_arm_order"]:
                captures[arm] = self._score_resident_arm_v19a(
                    arm, prepared, schedule_item, unit_scores, dense_commitments
                )
        finally:
            restore_hash = self._restore_and_verify_signed_wave_v19a()
        if tuple(captures) != tuple(schedule_item["resident_arm_order"]):
            raise RuntimeError("v19a signed-wave resident transaction changed")
        return restore_hash

    def estimate_disjoint_tier_attribution_v19a(self, seeds):
        seeds = [int(seed) for seed in seeds]
        schedule = resident_signed_wave_schedule_v19a(seeds)
        prepared, request_identity = self._prepared_fixed_batches_v19a()
        pre_probe = self._base_probe_v19a(prepared)
        unit_scores = {
            arm: np.full(
                (
                    10,
                    2,
                    32,
                    frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm],
                ),
                np.nan,
                dtype=np.float64,
            )
            for arm in mechanics_v19a.ARMS_V19A
        }
        dense_commitments = []
        restore_hashes = []
        for schedule_item in schedule:
            restore_hashes.append(self._run_signed_wave_v19a(
                schedule_item, prepared, unit_scores, dense_commitments
            ))
        if (
            any(not np.isfinite(values).all() for values in unit_scores.values())
            or len(dense_commitments) != 2560
            or len(restore_hashes) != 16
        ):
            raise RuntimeError("v19a disjoint-tier population capture is incomplete")
        boundary = self._population_boundary_audit_v4(0)
        unselected_audit, unselected_audit_sha256 = (
            _validate_population_boundary_v19a(
                boundary, self._v4_layer_plan_install
            )
        )
        post_probe = self._base_probe_v19a(prepared)
        if pre_probe != post_probe:
            raise RuntimeError("v19a pre/post base probes drifted")
        compact = mechanics_v19a.build_compact_estimator_summary_v19a(
            unit_scores, self._v19a_panel_bundle
        )
        unit_scores = None
        runtime_integrity = {
            "all_four_tp1_engines_every_signed_wave": True,
            "gpu_ids_zero_through_three_declared": True,
            "all_ten_panels_every_direction_sign_and_arm": True,
            "fixed_side_batch_identity_every_direction_sign_and_arm": True,
            "latin_arm_order_complete": True,
            "same_resident_perturbation_all_four_arms": True,
            "exact_reference_restored_once_per_signed_wave": True,
            "pre_post_base_probes_equal_all_four_arms": True,
            "population_boundary_audit_passed": True,
            "unselected_origin_audit_passed": unselected_audit["passed"],
            "tokenizer_and_prompt_logprob_contract_passed": True,
            "all_integrity_audits_passed": True,
        }
        summary = {
            "schema": "eggroll-es-disjoint-tier-attribution-summary-v19a",
            "experiment_name": EXPERIMENT_NAME_V19A,
            "alpha": 0.0,
            "sigma": 0.0003,
            "model_update_applied": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_applied": False,
            "frame_content_sha256": mechanics_v19a.FRAME_CERTIFICATE_CONTENT_SHA256_V19A,
            "perturbation_basis_sha256": prereg_v19a.PERTURBATION_BASIS_SHA256_V19A,
            "runtime_integrity": runtime_integrity,
            "arms": compact["arms"],
            "paired_bootstrap": compact["paired_bootstrap"],
            "persisted_response_vectors_or_row_content": False,
            "bootstrap_draws_persisted": False,
            "unit_scores_persisted": False,
        }
        summary["content_sha256_before_self_field"] = canonical_sha256(summary)
        gate = mechanics_v19a.evaluate_attribution_gate_v19a(summary)
        runtime_audit = {
            "schema": "eggroll-es-disjoint-tier-runtime-compact-audit-v19a",
            "fixed_request_identity_sha256": canonical_sha256(request_identity),
            "token_boundary_audit_sha256": self._v19a_token_boundary_audit_sha256,
            "pre_post_probe_identity_sha256": canonical_sha256({
                "pre": pre_probe, "post": post_probe
            }),
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "dense_result_commitments_sha256": canonical_sha256(dense_commitments),
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "unselected_origin_sha256": boundary["unselected_origin_sha256"],
            "unselected_origin_audit_sha256": unselected_audit_sha256,
            "signed_wave_count": 16,
            "panel_count": 10,
            "requests_per_engine_per_signed_wave": 990,
            "dense_result_commitment_count": 2560,
            "per_unit_scores_persisted": False,
            "bootstrap_replicates_persisted": False,
            "bootstrap_draws_persisted": False,
            "row_content_persisted": False,
        }
        runtime_audit["content_sha256_before_self_field"] = canonical_sha256(
            runtime_audit
        )
        return summary, gate, runtime_audit

    @staticmethod
    def _closed_surface_v19a(*_args, **_kwargs):
        raise RuntimeError(
            "v19a closes every update preparation commit checkpoint save and evaluation entrypoint"
        )

    configure_production_patch_compat_v18a = _closed_surface_v19a
    configure_train_panels_v13 = _closed_surface_v19a
    configure_anchor = _closed_surface_v19a
    estimate_production_patch_compat_v18a = _closed_surface_v19a
    estimate_train_panels_v13 = _closed_surface_v19a
    estimate_step_coefficients = _closed_surface_v19a
    apply_seed_coefficients = _closed_surface_v19a
    _evaluate_population_with_anchor = _closed_surface_v19a
    _persist_anchor_plan = _closed_surface_v19a
    _persist_identity_audit = _closed_surface_v19a
    _abort_update_v3 = _closed_surface_v19a
    _abort_update_v4 = _closed_surface_v19a
    train_step = _closed_surface_v19a
    evaluate_handle = _closed_surface_v19a
    evaluate_population_on_batch = _closed_surface_v19a
    eval_step = _closed_surface_v19a
    fit = _closed_surface_v19a


def load_runtime_trainer_v19a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    parent = anchor_v13.load_trainer(layer_bundle)

    class DisjointTierAttributionRuntimeTrainerV19A(
        DisjointTierAttributionRuntimeMixinV19A, parent
    ):
        pass

    return DisjointTierAttributionRuntimeTrainerV19A


def _make_trainer_v19a(layer_bundle):
    trainer_class = load_runtime_trainer_v19a(layer_bundle)
    return trainer_class(
        model_name=str(FROZEN_MODEL_V19A),
        checkpoint=None,
        sigma=0.0003,
        alpha=0.0,
        population_size=32,
        reward_shaping="z-scores",
        num_iterations=1,
        max_tokens=1,
        batch_size=250,
        mini_batch_size=250,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[],
        eval_dataloader_dict={},
        eval_freq=1,
        n_vllm_engines=4,
        n_gpu_per_vllm_engine=1,
        logging="none",
        global_seed=43,
        use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_V19A,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(FROZEN_OUTPUT_DIRECTORY_V19A),
    )


def _seal_v19a(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)


def _exclusive_write_json_v19a(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal_v19a(value)
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"v19a exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json_v19a(path, value):
    _seal_v19a(value)
    driver_v13.driver_v1.atomic_write_json(path, value)


def _new_launch_id_v19a():
    return f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"


def _claim_fresh_paths_v19a(attempt):
    root = (FROZEN_OUTPUT_DIRECTORY_V19A / EXPERIMENT_NAME_V19A).resolve()
    for _ in range(32):
        launch_id = _new_launch_id_v19a()
        attempt_path = root / "attempts" / f"{launch_id}.json"
        run_dir = root / "runs" / launch_id
        current = copy.deepcopy(attempt)
        current.update({"launch_id": launch_id, "run_directory": str(run_dir)})
        try:
            _exclusive_write_json_v19a(attempt_path, current)
        except FileExistsError:
            continue
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            current.update({
                "status": "failed",
                "phase": "run_directory_collision_after_attempt_claim",
                "failure_type": type(error).__name__,
                "failure_sha256": canonical_sha256(type(error).__name__),
            })
            _rewrite_json_v19a(attempt_path, current)
            continue
        return attempt_path, run_dir, current
    raise RuntimeError("v19a could not claim collision-safe fresh launch paths")


def _recursive_keys_v19a(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _recursive_keys_v19a(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys_v19a(item)


def _validate_compact_configuration_v19a(configuration):
    if (
        not isinstance(configuration, dict)
        or configuration.get("schema")
        != "eggroll-es-disjoint-tier-runtime-configuration-v19a"
        or configuration.get("panel_bundle_content_sha256")
        != mechanics_v19a.PANEL_BUNDLE_CONTENT_SHA256_V19A
        or configuration.get("worker_extension")
        != "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
        or configuration.get("engine_count") != 4
        or configuration.get("tp_per_engine") != 1
        or configuration.get("gpu_ids") != [0, 1, 2, 3]
        or configuration.get("model_update_allowed") is not False
        or configuration.get("checkpoint_write_allowed") is not False
        or configuration.get("evaluation_surfaces_opened") is not False
        or configuration.get("dataset_promotion_allowed") is not False
    ):
        raise RuntimeError("v19a compact configuration contract changed")
    prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(
        configuration
    )
    return configuration


def _validate_compact_runtime_audit_v19a(audit):
    if (
        not isinstance(audit, dict)
        or audit.get("schema")
        != "eggroll-es-disjoint-tier-runtime-compact-audit-v19a"
        or audit.get("signed_wave_count") != 16
        or audit.get("panel_count") != 10
        or audit.get("requests_per_engine_per_signed_wave") != 990
        or audit.get("dense_result_commitment_count") != 2560
        or audit.get("per_unit_scores_persisted") is not False
        or audit.get("bootstrap_replicates_persisted") is not False
        or audit.get("bootstrap_draws_persisted") is not False
        or audit.get("row_content_persisted") is not False
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v19a(audit))
    ):
        raise RuntimeError("v19a compact runtime audit contract changed")
    prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(audit)
    return audit


def validate_compact_report_v19a(
    report, *, expected_recipe, expected_implementation,
):
    if (
        not isinstance(report, dict)
        or set(report)
        != {
            "schema",
            "recipe",
            "configuration",
            "runtime_audit",
            "summary",
            "gate",
            "implementation",
            "model_update_applied",
            "checkpoint_written",
            "evaluation_surfaces_opened",
            "dataset_promotion_applied",
            "content_sha256_before_self_field",
        }
        or report.get("schema")
        != "eggroll-es-disjoint-tier-attribution-report-v19a"
        or report.get("model_update_applied") is not False
        or report.get("checkpoint_written") is not False
        or report.get("evaluation_surfaces_opened") is not False
        or report.get("dataset_promotion_applied") is not False
        or report.get("recipe") != expected_recipe
        or report.get("implementation") != expected_implementation
        or report.get("recipe", {}).get("implementation_bundle_sha256")
        != report.get("implementation", {}).get("bundle_sha256")
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v19a(report))
        or report.get("summary", {}).get(
            "persisted_response_vectors_or_row_content"
        )
        is not False
        or report.get("gate")
        != mechanics_v19a.evaluate_attribution_gate_v19a(report.get("summary"))
        or report.get("gate", {}).get("dataset_promotion_authorized") is not False
        or report.get("gate", {}).get("model_update_authorized") is not False
        or report.get("gate", {}).get("evaluation_authorized") is not False
        or any(
            str(key).lower() in FORBIDDEN_PERSISTED_KEYS_V19A
            for key in _recursive_keys_v19a(report)
        )
    ):
        raise RuntimeError("v19a compact report contract changed")
    _validate_compact_configuration_v19a(report["configuration"])
    _validate_compact_runtime_audit_v19a(report["runtime_audit"])
    prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(report)
    return report


def run_exact_v19a(layer_bundle, panel_bundle, implementation, recipe):
    if _validate_moe_backend_environment_v19a() != recipe.get("moe_backend"):
        raise ValueError("v19a runtime MoE backend differs from sealed recipe")
    provenance = _source_provenance_v19a(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v19a",
        "status": "launching",
        "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V19A,
        "recipe": recipe,
        "source_provenance": provenance,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
    }
    attempt_path, run_dir, attempt = _claim_fresh_paths_v19a(attempt)
    report_path = run_dir / REPORT_NAME_V19A
    trainer = None
    failure = None
    configuration = summary = gate = runtime_audit = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v19a(layer_bundle)
        configuration = trainer.configure_disjoint_tier_attribution_v19a(
            panel_bundle, frozen_layer_plan=layer_bundle
        )
        summary, gate, runtime_audit = (
            trainer.estimate_disjoint_tier_attribution_v19a(
                prereg_v19a.PERTURBATION_SEEDS_V19A
            )
        )
        if gate != mechanics_v19a.evaluate_attribution_gate_v19a(summary):
            raise RuntimeError("v19a hardened attribution gate changed")
    except BaseException as error:
        failure = error
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "inside_disjoint_tier_train_only_runtime",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure)
            }),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_applied": False,
        })
        _rewrite_json_v19a(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-disjoint-tier-attribution-report-v19a",
        "recipe": recipe,
        "configuration": configuration,
        "runtime_audit": runtime_audit,
        "summary": summary,
        "gate": gate,
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    try:
        validate_compact_report_v19a(
            report, expected_recipe=recipe, expected_implementation=implementation
        )
        _exclusive_write_json_v19a(report_path, report)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "validating_or_writing_compact_report",
            "failure_type": type(error).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(error).__name__, "repr": repr(error)
            }),
            "report_exists_after_attempt": report_path.exists(),
        })
        _rewrite_json_v19a(attempt_path, attempt)
        raise
    attempt.update({
        "status": "complete",
        "phase": "after_trainer_cleanup_and_compact_report",
        "report_exists_after_attempt": True,
        "report_binding": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
    })
    _rewrite_json_v19a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v19a(argv)
    moe_backend = _validate_moe_backend_environment_v19a()
    layer_bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v19a().parse_args(remaining)
    implementation = implementation_identity_v19a()
    preregistration, panel_bundle = _load_bound_inputs_v19a(layer_bundle)
    recipe = recipe_v19a(
        layer_bundle, preregistration, panel_bundle, implementation, moe_backend
    )
    validate_runtime_v19a(args, layer_bundle, implementation, recipe)
    if args.v19a_dry_run:
        payload = {
            "schema": "eggroll-es-disjoint-tier-attribution-dry-run-v19a",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "offline_phase_bundle_sha256": implementation[
                "offline_phase_bundle_sha256"
            ],
            "trainer_phase_bundle_sha256": implementation[
                "trainer_phase_bundle_sha256"
            ],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "implementation": implementation,
            "recipe": recipe,
            "real_launch_requires_committed_source_bundle": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = canonical_sha256(payload)
        prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(
            payload
        )
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v19a(layer_bundle, panel_bundle, implementation, recipe)


if __name__ == "__main__":
    main()
