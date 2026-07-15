#!/usr/bin/env python3
"""Fail-closed train-only runtime for V17A paired data compatibility."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import eggroll_es_paired_data_compat_preregistration_v17a as prereg_v17a
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_paired_data_compat_v17a as mechanics_v17a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V17A = driver_v13.FROZEN_MODEL_V13
FROZEN_OUTPUT_DIRECTORY_V17A = driver_v13.FROZEN_OUTPUT_DIRECTORY_V13
EXPERIMENT_NAME_V17A = prereg_v17a.EXPERIMENT_NAME_V17A
REPORT_NAME_V17A = "paired_data_compat_v17a.json"
TEST_PATH_V17A = (
    ROOT / "test_eggroll_es_paired_data_compat_runtime_v17a.py"
).resolve()
TRAINER_PATH_V17A = Path(mechanics_v17a.__file__).resolve()
TRAINER_TEST_PATH_V17A = (
    ROOT / "test_eggroll_es_paired_data_compat_trainer_v17a.py"
).resolve()
BASE_SPECIALIST_PATH_V17A = Path(base.__file__).resolve()

TRAINER_PHASE_HASHES_V17A = {
    "trainer_mechanics_v17a": (
        "61950d9431cbe222f4cfcc24853dd99aa4e3928180fb14b9ca48169bedb178f1"
    ),
    "trainer_tests_v17a": (
        "eed2d1f597d1723039c2f587c8a57e6ffba49b587376e15e45edf2ca6df60280"
    ),
}
TRAINER_PHASE_PATHS_V17A = {
    "trainer_mechanics_v17a": TRAINER_PATH_V17A,
    "trainer_tests_v17a": TRAINER_TEST_PATH_V17A,
}
FROZEN_PHASE_HASHES_V17A = {
    "prereg_module_v17a": (
        "c763b5b347ae28f717241efecf3604f35e5cf1b8c1bf742d6c7d64a2486e1781"
    ),
    "prereg_tests_v17a": (
        "a412aa1be7ced18c7b084efb67a702da7346d2ae770244c575d8c1088f1222f0"
    ),
    "preregistration_v17a": (
        "85a30be591f72376e220447ce9f1be0d04919b2855a987b757d0d71bd90fba1f"
    ),
    "protocol_v17a": (
        "cd1194cc3799475c438e649e6514c5fb9e9814f679fff27c1497d0ecca176cac"
    ),
    "frame_builder_v17a": prereg_v17a.FRAME_BUILDER_FILE_SHA256_V17A,
    "frame_tests_v17a": prereg_v17a.FRAME_TEST_FILE_SHA256_V17A,
    "frame_manifest_v17a": prereg_v17a.FRAME_FILE_SHA256_V17A,
    "base_specialist_v17a": (
        "bbffbf16747ec514c67e48daab696560eb3309f5a3edf0a700257969cad35c23"
    ),
}
FROZEN_PHASE_PATHS_V17A = {
    "prereg_module_v17a": Path(prereg_v17a.__file__).resolve(),
    "prereg_tests_v17a": (
        ROOT / "test_eggroll_es_paired_data_compat_preregistration_v17a.py"
    ).resolve(),
    "preregistration_v17a": prereg_v17a.PREREGISTRATION_PATH_V17A,
    "protocol_v17a": prereg_v17a.PROTOCOL_PATH_V17A,
    "frame_builder_v17a": (
        ROOT / "build_eggroll_es_joint_panels_v17a.py"
    ).resolve(),
    "frame_tests_v17a": (
        ROOT / "test_eggroll_es_joint_panels_v17a.py"
    ).resolve(),
    "frame_manifest_v17a": prereg_v17a.FRAME_PATH_V17A,
    "base_specialist_v17a": BASE_SPECIALIST_PATH_V17A,
}
IMPLEMENTATION_PATHS_V17A = {
    **driver_v13.IMPLEMENTATION_PATHS_V13,
    **FROZEN_PHASE_PATHS_V17A,
    **TRAINER_PHASE_PATHS_V17A,
    "runtime_driver_v17a": Path(__file__).resolve(),
    "runtime_tests_v17a": TEST_PATH_V17A,
}
FORBIDDEN_SURFACE_TOKENS_V17A = (
    "checkpoint", "update", "heldout", "validation", "ood", "benchmark",
    "eval", "save-model", "train-dataset",
)
FORBIDDEN_PERSISTED_KEYS_V17A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "unit_scores", "responses", "coefficients",
    "bootstrap_replicates", "bootstrap_draws", "row_content",
}
MOE_OVERRIDE_ENVIRONMENT_V17A = (
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)

canonical_sha256 = prereg_v17a.canonical_sha256
file_sha256 = prereg_v17a.file_sha256
anchor_v4 = anchor_v13.anchor_v4


def _without_self_v17a(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _assert_train_only_argv_v17a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V17A):
            raise ValueError(f"v17a rejects forbidden runtime surface: {token}")


def implementation_identity_v17a():
    inherited = driver_v13.implementation_identity_v13()
    mechanics_v17a._load_hardened_preregistration_v17a()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V17A.items()
    }
    if {key: files[key] for key in inherited["files"]} != inherited["files"]:
        raise RuntimeError("v17a exact inherited V13/V4 implementation changed")
    expected = {**FROZEN_PHASE_HASHES_V17A, **TRAINER_PHASE_HASHES_V17A}
    if {key: files[key]["file_sha256"] for key in expected} != expected:
        raise RuntimeError("v17a frozen preregistration/trainer phase changed")
    trainer_phase = {
        key: files[key] for key in TRAINER_PHASE_PATHS_V17A
    }
    return {
        "files": files,
        "inherited_v13_bundle_sha256": inherited["bundle_sha256"],
        "trainer_phase_bundle_sha256": canonical_sha256(trainer_phase),
        "bundle_sha256": canonical_sha256(files),
    }


def _source_provenance_v17a(implementation):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"v17a real launch requires committed source: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v17a source differs from HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    result = {
        "schema": "eggroll-es-committed-source-bundle-v17a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _parser_v17a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def _declared_moe_backend_v17a():
    return {
        "moe_backend": "default_triton",
        "override_environment": {
            name: None for name in MOE_OVERRIDE_ENVIRONMENT_V17A
        },
        "v16_task_ab_decision_retained": "default_triton",
    }


def _validate_moe_backend_environment_v17a():
    conflicts = {
        name: os.environ.get(name)
        for name in MOE_OVERRIDE_ENVIRONMENT_V17A
        if os.environ.get(name) not in (None, "")
    }
    if conflicts:
        raise ValueError("v17a requires every MoE backend override unset")
    return _declared_moe_backend_v17a()


def _load_bound_inputs_v17a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    plan = anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    preregistration = mechanics_v17a._load_hardened_preregistration_v17a()
    panel_bundle = mechanics_v17a.load_paired_panel_bundle_v17a()
    if (
        layer_bundle["plan_sha256"]
        != driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
        or Path(layer_bundle["path"]).resolve() != Path(plan["path"]).resolve()
        or layer_bundle["file_sha256"] != plan["file_sha256"]
        or layer_bundle["model_config_sha256"]
        != prereg_v17a.MODEL_CONFIG_SHA256_V17A
        or Path(preregistration["frozen_recipe"]["model"]).resolve()
        != FROZEN_MODEL_V17A
        or preregistration["frozen_recipe"]["alpha"] != 0.0
        or preregistration["frozen_recipe"]["sigma"] != 0.0003
        or preregistration["frozen_recipe"]["population_size"] != 32
        or preregistration["frozen_recipe"]["layer_plan"]["plan_sha256"]
        != layer_bundle["plan_sha256"]
        or preregistration["scoring"]["dense_reward_config_sha256"]
        != anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
    ):
        raise RuntimeError("v17a model/layer/scoring preregistration changed")
    return preregistration, panel_bundle


def recipe_v17a(
    layer_bundle, preregistration, panel_bundle, implementation, moe_backend,
):
    if moe_backend != _declared_moe_backend_v17a():
        raise RuntimeError("v17a default Triton backend declaration changed")
    panel_contract = {}
    for panel_name in mechanics_v17a.PANEL_NAMES_V17A:
        panel = panel_bundle["panels"][panel_name]
        panel_contract[panel_name] = {
            "role": panel["role"],
            "rows_per_arm": len(panel["unit_ids"]),
            "ordered_unit_identity_sha256": panel[
                "ordered_unit_identity_sha256"
            ],
            "ordered_arm_row_identities": {
                arm: {
                    "row_identity_sha256": panel["arms"][arm][
                        "ordered_row_identity_sha256"
                    ],
                }
                for arm in mechanics_v17a.ARMS_V17A
            },
        }
    recipe = {
        "schema": "eggroll-es-paired-data-compat-recipe-v17a",
        "experiment_name": EXPERIMENT_NAME_V17A,
        "model": str(FROZEN_MODEL_V17A),
        "preregistration": copy.deepcopy(panel_bundle["preregistration"]),
        "frame": copy.deepcopy(panel_bundle["frame"]),
        "sources": copy.deepcopy(panel_bundle["sources"]),
        "materialized_panel_bundle_content_sha256": panel_bundle[
            "content_sha256_before_self_field"
        ],
        "panels": panel_contract,
        "layer_plan": {
            "path": layer_bundle["path"],
            "file_sha256": layer_bundle["file_sha256"],
            "plan_sha256": layer_bundle["plan_sha256"],
            "model_config_sha256": layer_bundle["model_config_sha256"],
        },
        "perturbation": {
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 32,
            "basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
            "basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
            "seed_sha256": canonical_sha256(anchor_v13.PERTURBATION_SEEDS_V13),
            "signed_wave_schedule_sha256": canonical_sha256(
                mechanics_v17a.resident_signed_wave_schedule_v17a()
            ),
            "same_resident_perturbation_both_arms": True,
            "restore_once_after_both_arms": True,
        },
        "scoring": {
            "objective": "per_example_mean_full_gold_answer_token_logprob",
            "dense_reward_config_sha256": (
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
            ),
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
            "fixed_prompts_materialized_once": True,
        },
        "bootstrap": copy.deepcopy(preregistration["analysis"]["bootstrap"]),
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
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V17A),
    }
    recipe["content_sha256_before_self_field"] = canonical_sha256(recipe)
    prereg_v17a.validate_persisted_sha256_fields_v17a(recipe)
    return recipe


def validate_runtime_v17a(args, layer_bundle, implementation, recipe):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    if (
        not isinstance(args.v17a_dry_run, bool)
        or recipe.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v17a(recipe))
        or recipe.get("model_update_allowed") is not False
        or recipe.get("checkpoint_write_allowed") is not False
        or recipe.get("evaluation_surfaces_opened") is not False
        or recipe.get("moe_backend") != _declared_moe_backend_v17a()
        or recipe.get("hardware") != {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        }
    ):
        raise ValueError("v17a frozen train-only runtime recipe changed")
    if not args.v17a_dry_run and (
        args.expected_implementation_bundle_sha256 is None
        or args.expected_recipe_sha256 is None
    ):
        raise ValueError("v17a real launch requires implementation and recipe hashes")
    if (
        args.expected_implementation_bundle_sha256 is not None
        and args.expected_implementation_bundle_sha256
        != implementation["bundle_sha256"]
    ):
        raise ValueError("v17a implementation bundle hash changed")
    if (
        args.expected_recipe_sha256 is not None
        and args.expected_recipe_sha256
        != recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v17a recipe hash changed")


def _request_identity_v17a(prompt_items):
    return canonical_sha256([
        item["prompt_token_ids"] for item in prompt_items
    ])


def _score_panel_unit_outputs_v17a(dense_items, outputs):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = np.asarray([
        item["mean_answer_token_logprob"] for item in dense["examples"]
    ], dtype=np.float64)
    if (
        rewards.shape != (mechanics_v17a.frame_v17a.PANEL_SIZE_V17A,)
        or not np.isfinite(rewards).all()
    ):
        raise RuntimeError("v17a per-unit dense score coverage changed")
    return rewards, canonical_sha256(dense)


class PairedDataCompatRuntimeMixinV17A:
    """One-trainer resident runtime; all historical update/eval paths close."""

    def configure_paired_data_compat_v17a(
        self, panel_bundle, *, frozen_layer_plan,
    ):
        panel_bundle = mechanics_v17a.validate_paired_panel_bundle_v17a(
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
            raise ValueError("v17a requires one four-TP1-engine alpha-zero trainer")
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
            reports, frozen_layer_plan,
        )
        self._v4_layer_plan = frozen_layer_plan
        self._v4_layer_plan_install = install
        self._v4_reward_config = dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4)
        self._v4_reward_config_sha256 = (
            anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        )
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v17a workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4",
            (4, "exact_reference"),
        )
        reference = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(reference, fresh=True)
        self._v17a_panel_bundle = panel_bundle
        return {
            "schema": "eggroll-es-paired-runtime-configuration-v17a",
            "layer_plan_install_sha256": canonical_sha256(install),
            "reference_identity_sha256": canonical_sha256(
                reference["reference_identity"]
            ),
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "worker_extension": (
                "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
            ),
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
        }

    def _prepared_fixed_batches_v17a(self):
        bundle = mechanics_v17a.validate_paired_panel_bundle_v17a(
            self._v17a_panel_bundle
        )
        prepared = {}
        request_identity = {}
        for arm in mechanics_v17a.ARMS_V17A:
            panels = {}
            prompt_items = []
            cursor = 0
            for panel_name in mechanics_v17a.PANEL_NAMES_V17A:
                arm_batch = bundle["panels"][panel_name]["arms"][arm]
                prompts = [base.specialist_template(item) for item in arm_batch["questions"]]
                templated_identity = canonical_sha256({
                    "prompts": prompts, "answers": arm_batch["answers"],
                })
                dense_items = anchor_v4.prepare_gold_answer_items_v4(
                    self.tokenizer, prompts, arm_batch["answers"],
                )
                current_items = [
                    {"prompt_token_ids": item["prompt_token_ids"]}
                    for item in dense_items
                ]
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(current_items)),
                    "templated_prompt_answer_sha256": templated_identity,
                }
                prompt_items.extend(current_items)
                cursor += len(current_items)
            if cursor != 190:
                raise RuntimeError("v17a fixed arm request count changed")
            prepared[arm] = {"panels": panels, "prompt_items": prompt_items}
            request_identity[arm] = _request_identity_v17a(prompt_items)
        self._v17a_fixed_request_identity = request_identity
        return prepared, request_identity

    def _assert_fixed_request_v17a(self, arm, prepared):
        if (
            arm not in mechanics_v17a.ARMS_V17A
            or _request_identity_v17a(prepared[arm]["prompt_items"])
            != self._v17a_fixed_request_identity[arm]
        ):
            raise RuntimeError("v17a fixed request identity changed before generation")

    def _base_probe_v17a(self, prepared):
        result = {}
        for arm in mechanics_v17a.ARMS_V17A:
            self._assert_fixed_request_v17a(arm, prepared)
            outputs = anchor_v13.anchor_v11.anchor_v1.dispatch_eval_batch(
                self.engines,
                list(prepared[arm]["prompt_items"]),
                self._dense_sampling_params_v4(0),
                self._resolve,
            )
            panel_hashes = {}
            for panel_name in mechanics_v17a.PANEL_NAMES_V17A:
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                _rewards, dense_hash = _score_panel_unit_outputs_v17a(
                    panel["dense_items"], outputs[start:end],
                )
                panel_hashes[panel_name] = dense_hash
            result[arm] = canonical_sha256(panel_hashes)
        return result

    def _perturb_signed_wave_v17a(self, engine_seeds, negate):
        if len(engine_seeds) != 4:
            raise RuntimeError("v17a perturbation wave is partial")
        self._resolve([
            self.engines[index].collective_rpc.remote(
                "perturb_self_weights",
                args=(int(seed), 0.0003, bool(negate)),
            )
            for index, seed in enumerate(engine_seeds)
        ])

    def _score_resident_arm_v17a(
        self, arm, prepared, schedule_item, unit_scores, dense_commitments,
    ):
        self._assert_fixed_request_v17a(arm, prepared)
        batches = self._resolve([
            engine.generate.remote(
                list(prepared[arm]["prompt_items"]),
                self._dense_sampling_params_v4(0),
                use_tqdm=False,
            )
            for engine in self.engines
        ])
        if (
            not isinstance(batches, list)
            or len(batches) != 4
            or any(len(batch) != 190 for batch in batches)
        ):
            raise RuntimeError("v17a resident arm generation is incomplete")
        sign_index = mechanics_v17a.SIGNS_V17A.index(schedule_item["sign"])
        arm_index = mechanics_v17a.ARMS_V17A.index(arm)
        for engine_index, batch in enumerate(batches):
            direction_index = 4 * schedule_item["population_wave_index"] + engine_index
            for panel_index, panel_name in enumerate(
                mechanics_v17a.PANEL_NAMES_V17A
            ):
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                rewards, dense_hash = _score_panel_unit_outputs_v17a(
                    panel["dense_items"], batch[start:end],
                )
                unit_scores[
                    arm_index, panel_index, sign_index, direction_index,
                ] = rewards
                dense_commitments.append(dense_hash)
        return canonical_sha256(dense_commitments[-20:])

    def _restore_and_verify_signed_wave_v17a(self):
        self._restore_all_engines_exact()
        checks = self._resolve([
            engine.collective_rpc.remote(
                "verify_self_exact_reference", args=(),
            )
            for engine in self.engines
        ])
        if not anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks,
            lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v17a signed-wave exact reference restore failed")
        return canonical_sha256(checks)

    def _run_signed_wave_v17a(
        self, schedule_item, prepared, unit_scores, dense_commitments,
    ):
        restore_hashes = []
        captures = mechanics_v17a.execute_paired_resident_signed_wave_v17a(
            schedule_item,
            perturb=lambda seeds, negate: self._perturb_signed_wave_v17a(
                seeds, negate,
            ),
            score_arm=lambda arm: self._score_resident_arm_v17a(
                arm, prepared, schedule_item, unit_scores, dense_commitments,
            ),
            restore=lambda: restore_hashes.append(
                self._restore_and_verify_signed_wave_v17a()
            ),
        )
        if len(restore_hashes) != 1 or tuple(captures) != tuple(
            schedule_item["resident_arm_order"]
        ):
            raise RuntimeError("v17a signed-wave resident transaction changed")
        return restore_hashes[0]

    def estimate_paired_data_compat_v17a(self, seeds):
        seeds = [int(seed) for seed in seeds]
        schedule = mechanics_v17a.resident_signed_wave_schedule_v17a(seeds)
        prepared, request_identity = self._prepared_fixed_batches_v17a()
        pre_probe = self._base_probe_v17a(prepared)
        unit_scores = np.full((2, 5, 2, 32, 38), np.nan, dtype=np.float64)
        dense_commitments = []
        restore_hashes = []
        for schedule_item in schedule:
            restore_hashes.append(self._run_signed_wave_v17a(
                schedule_item, prepared, unit_scores, dense_commitments,
            ))
        if (
            not np.isfinite(unit_scores).all()
            or len(dense_commitments) != 2 * 5 * 2 * 32
            or len(restore_hashes) != 16
        ):
            raise RuntimeError("v17a paired population capture is incomplete")
        boundary = self._population_boundary_audit_v4(0)
        if (
            boundary.get("passed") is not True
            or boundary.get("audit_sha256")
            != canonical_sha256({
                key: value for key, value in boundary.items()
                if key != "audit_sha256"
            })
        ):
            raise RuntimeError("v17a population boundary audit failed")
        post_probe = self._base_probe_v17a(prepared)
        if pre_probe != post_probe:
            raise RuntimeError("v17a paired pre/post base probes drifted")
        compact = mechanics_v17a.build_compact_estimator_summary_v17a(
            unit_scores, self._v17a_panel_bundle,
        )
        unit_scores = None
        runtime_integrity = {
            "all_four_engines_every_signed_wave": True,
            "fixed_side_batch_identity_every_direction_and_sign": True,
            "same_resident_perturbation_both_versions": True,
            "alternating_version_order_complete": True,
            "exact_reference_restoration_passed": True,
            "pre_post_base_probes_equal_both_versions": True,
            "population_boundary_audit_passed": True,
            "tokenizer_and_prompt_logprob_contract_passed": True,
            "all_integrity_audits_passed": True,
        }
        summary = {
            "schema": "eggroll-es-paired-data-compat-summary-v17a",
            "experiment_name": EXPERIMENT_NAME_V17A,
            "alpha": 0.0,
            "sigma": 0.0003,
            "model_update_applied": False,
            "evaluation_surfaces_opened": False,
            "frame_content_sha256": prereg_v17a.FRAME_CONTENT_SHA256_V17A,
            "perturbation_basis_sha256": (
                anchor_v13.PERTURBATION_BASIS_SHA256_V13
            ),
            "runtime_integrity": runtime_integrity,
            "versions": compact["versions"],
            "paired_bootstrap": compact["paired_bootstrap"],
            "cross_dataset_direction_similarity_diagnostic": compact[
                "cross_dataset_direction_similarity_diagnostic"
            ],
            "persisted_response_vectors_or_row_content": False,
        }
        summary["content_sha256_before_self_field"] = canonical_sha256(summary)
        gate = prereg_v17a.evaluate_candidate_v17a(summary)
        runtime_audit = {
            "schema": "eggroll-es-paired-runtime-compact-audit-v17a",
            "fixed_request_identity_sha256": canonical_sha256(request_identity),
            "pre_post_probe_identity_sha256": canonical_sha256({
                "pre": pre_probe, "post": post_probe,
            }),
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "dense_result_commitments_sha256": canonical_sha256(
                dense_commitments
            ),
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "signed_wave_count": 16,
            "per_unit_scores_persisted": False,
            "bootstrap_replicates_persisted": False,
        }
        runtime_audit["content_sha256_before_self_field"] = canonical_sha256(
            runtime_audit
        )
        return summary, gate, runtime_audit

    @staticmethod
    def _closed_surface_v17a(*_args, **_kwargs):
        raise RuntimeError("v17a closes every update checkpoint and eval entrypoint")

    configure_train_panels_v13 = _closed_surface_v17a
    estimate_train_panels_v13 = _closed_surface_v17a
    estimate_step_coefficients = _closed_surface_v17a
    apply_seed_coefficients = _closed_surface_v17a
    train_step = _closed_surface_v17a
    evaluate_handle = _closed_surface_v17a
    evaluate_population_on_batch = _closed_surface_v17a
    eval_step = _closed_surface_v17a
    fit = _closed_surface_v17a


def load_runtime_trainer_v17a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    parent = anchor_v13.load_trainer(layer_bundle)

    class PairedDataCompatRuntimeTrainerV17A(
        PairedDataCompatRuntimeMixinV17A, parent,
    ):
        pass

    return PairedDataCompatRuntimeTrainerV17A


def _make_trainer_v17a(layer_bundle):
    trainer_class = load_runtime_trainer_v17a(layer_bundle)
    return trainer_class(
        model_name=str(FROZEN_MODEL_V17A), checkpoint=None,
        sigma=0.0003, alpha=0.0, population_size=32,
        reward_shaping="z-scores", num_iterations=1, max_tokens=1,
        batch_size=190, mini_batch_size=190,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_V17A,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(FROZEN_OUTPUT_DIRECTORY_V17A),
    )


def _seal_v17a(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)


def _exclusive_write_json_v17a(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal_v17a(value)
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"v17a exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json_v17a(path, value):
    _seal_v17a(value)
    driver_v13.driver_v1.atomic_write_json(path, value)


def _attempt_path_v17a():
    return FROZEN_OUTPUT_DIRECTORY_V17A / f".{EXPERIMENT_NAME_V17A}.launch_attempt.json"


def _recursive_keys_v17a(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _recursive_keys_v17a(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys_v17a(item)


def _validate_compact_configuration_v17a(configuration):
    if (
        not isinstance(configuration, dict)
        or set(configuration) != {
            "schema", "layer_plan_install_sha256", "reference_identity_sha256",
            "panel_bundle_content_sha256", "worker_extension",
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_surfaces_opened",
        }
        or configuration.get("schema")
        != "eggroll-es-paired-runtime-configuration-v17a"
        or configuration.get("panel_bundle_content_sha256")
        != mechanics_v17a.PAIRED_PANEL_BUNDLE_CONTENT_SHA256_V17A
        or configuration.get("worker_extension")
        != "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
        or configuration.get("model_update_allowed") is not False
        or configuration.get("checkpoint_write_allowed") is not False
        or configuration.get("evaluation_surfaces_opened") is not False
    ):
        raise RuntimeError("v17a compact configuration contract changed")
    return configuration


def _validate_compact_runtime_audit_v17a(audit):
    if (
        not isinstance(audit, dict)
        or set(audit) != {
            "schema", "fixed_request_identity_sha256",
            "pre_post_probe_identity_sha256", "signed_wave_schedule_sha256",
            "restore_checks_sha256", "dense_result_commitments_sha256",
            "population_boundary_audit_sha256", "signed_wave_count",
            "per_unit_scores_persisted", "bootstrap_replicates_persisted",
            "content_sha256_before_self_field",
        }
        or audit.get("schema")
        != "eggroll-es-paired-runtime-compact-audit-v17a"
        or audit.get("signed_wave_count") != 16
        or audit.get("per_unit_scores_persisted") is not False
        or audit.get("bootstrap_replicates_persisted") is not False
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v17a(audit))
    ):
        raise RuntimeError("v17a compact runtime audit contract changed")
    return audit


def validate_compact_report_v17a(
    report, *, expected_recipe, expected_implementation,
):
    if (
        not isinstance(report, dict)
        or set(report) != {
            "schema", "recipe", "configuration", "runtime_audit", "summary",
            "gate", "implementation", "model_update_applied",
            "checkpoint_written", "evaluation_surfaces_opened",
            "content_sha256_before_self_field",
        }
        or report.get("schema") != "eggroll-es-paired-data-compat-report-v17a"
        or report.get("model_update_applied") is not False
        or report.get("checkpoint_written") is not False
        or report.get("evaluation_surfaces_opened") is not False
        or report.get("recipe") != expected_recipe
        or report.get("implementation") != expected_implementation
        or report.get("recipe", {}).get("implementation_bundle_sha256")
        != report.get("implementation", {}).get("bundle_sha256")
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v17a(report))
        or report.get("summary", {}).get(
            "persisted_response_vectors_or_row_content"
        ) is not False
        or report.get("gate")
        != prereg_v17a.evaluate_candidate_v17a(report.get("summary"))
        or any(
            str(key).lower() in FORBIDDEN_PERSISTED_KEYS_V17A
            for key in _recursive_keys_v17a(report)
        )
    ):
        raise RuntimeError("v17a compact report contract changed")
    _validate_compact_configuration_v17a(report["configuration"])
    _validate_compact_runtime_audit_v17a(report["runtime_audit"])
    prereg_v17a.validate_persisted_sha256_fields_v17a(report)
    return report


def run_exact_v17a(
    layer_bundle, panel_bundle, implementation, recipe,
):
    if (
        _validate_moe_backend_environment_v17a()
        != recipe.get("moe_backend")
    ):
        raise ValueError("v17a runtime MoE backend differs from sealed recipe")
    attempt_path = _attempt_path_v17a().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V17A / EXPERIMENT_NAME_V17A).resolve()
    report_path = run_dir / REPORT_NAME_V17A
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v17a requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v17a(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v17a",
        "status": "launching",
        "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V17A,
        "run_directory": str(run_dir),
        "recipe": recipe,
        "source_provenance": provenance,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
    }
    _exclusive_write_json_v17a(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({
            "status": "failed",
            "phase": "existing_run_after_attempt_claim",
            "failure_type": "FreshRunReservationError",
            "failure_sha256": canonical_sha256(
                "v17a run directory appeared after exclusive claim"
            ),
        })
        _rewrite_json_v17a(attempt_path, attempt)
        raise ValueError("v17a run directory appeared after exclusive claim")
    trainer = None
    failure = None
    configuration = summary = gate = runtime_audit = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v17a(layer_bundle)
        configuration = trainer.configure_paired_data_compat_v17a(
            panel_bundle, frozen_layer_plan=layer_bundle,
        )
        summary, gate, runtime_audit = trainer.estimate_paired_data_compat_v17a(
            anchor_v13.PERTURBATION_SEEDS_V13,
        )
        if gate != prereg_v17a.evaluate_candidate_v17a(summary):
            raise RuntimeError("v17a hardened gate recomputation changed")
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
            "phase": "inside_paired_train_only_runtime",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
        })
        _rewrite_json_v17a(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-paired-data-compat-report-v17a",
        "recipe": recipe,
        "configuration": configuration,
        "runtime_audit": runtime_audit,
        "summary": summary,
        "gate": gate,
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    validate_compact_report_v17a(
        report,
        expected_recipe=recipe,
        expected_implementation=implementation,
    )
    try:
        _exclusive_write_json_v17a(report_path, report)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "writing_compact_report",
            "failure_type": type(error).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(error).__name__, "repr": repr(error),
            }),
            "report_exists_after_attempt": report_path.exists(),
        })
        _rewrite_json_v17a(attempt_path, attempt)
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
    _rewrite_json_v17a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v17a(argv)
    moe_backend = _validate_moe_backend_environment_v17a()
    layer_bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v17a().parse_args(remaining)
    implementation = implementation_identity_v17a()
    preregistration, panel_bundle = _load_bound_inputs_v17a(layer_bundle)
    recipe = recipe_v17a(
        layer_bundle, preregistration, panel_bundle, implementation,
        moe_backend,
    )
    validate_runtime_v17a(args, layer_bundle, implementation, recipe)
    if args.v17a_dry_run:
        payload = {
            "schema": "eggroll-es-paired-data-compat-dry-run-v17a",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
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
        prereg_v17a.validate_persisted_sha256_fields_v17a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v17a(
        layer_bundle, panel_bundle, implementation, recipe,
    )


if __name__ == "__main__":
    main()
