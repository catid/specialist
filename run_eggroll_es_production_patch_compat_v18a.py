#!/usr/bin/env python3
"""Fail-closed train-only runtime for the V18A production patch scan."""

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

import eggroll_es_production_overlay_scan_preregistration_v18a as prereg_v18a
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_production_patch_compat_v18a as mechanics_v18a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V18A = driver_v13.FROZEN_MODEL_V13
FROZEN_OUTPUT_DIRECTORY_V18A = driver_v13.FROZEN_OUTPUT_DIRECTORY_V13
EXPERIMENT_NAME_V18A = prereg_v18a.EXPERIMENT_NAME_V18A
REPORT_NAME_V18A = "production_patch_compat_v18a.json"
TEST_PATH_V18A = (
    ROOT / "test_run_eggroll_es_production_patch_compat_v18a.py"
).resolve()
TRAINER_PATH_V18A = Path(mechanics_v18a.__file__).resolve()
TRAINER_TEST_PATH_V18A = (
    ROOT / "test_train_eggroll_es_production_patch_compat_v18a.py"
).resolve()

BASE_FRAME_COMMIT_V18A = "7055a62d67d030cfddf594c9ecf1d9e290f9d0d5"
TOKEN_CORRECTION_COMMIT_V18A = "a3480e5535d252dbc3dec69e0657876f5fb4b39b"
RUNTIME_CORRECTION_COMMIT_V18A = (
    "3b7762215280be2d2bec2c63ec29a58ff7aadc6d"
)
TRAINER_COMMIT_V18A = "5f511f42bcd79a708d807b5dbe151181ce75ae2c"

TRAINER_PHASE_HASHES_V18A = {
    "trainer_mechanics_v18a": (
        "6741ec2a89ff4f1e595c648dc6f79166caa5495e16b9f97928152ac5fb7bb277"
    ),
    "trainer_tests_v18a": (
        "d3649d9c4829888796c83a760a5108f1f64b92563a30da0fa1ac30293d38212b"
    ),
}
TRAINER_PHASE_PATHS_V18A = {
    "trainer_mechanics_v18a": TRAINER_PATH_V18A,
    "trainer_tests_v18a": TRAINER_TEST_PATH_V18A,
}
FROZEN_PHASE_HASHES_V18A = {
    "frame_builder_v18a": (
        "73d7c71233946996d64d776fb68c31609ad512379a15a75fae46dd0d8c395ba0"
    ),
    "frame_tests_v18a": (
        "2097757d0f10b145387c42985e563bd94e2194ff5262ae538d16072fcb75788d"
    ),
    "frame_certificate_v18a": (
        "0887f936fd00d205fab46d490810732d16ddc2b34d96fdc507d43c20dec60f8e"
    ),
    "token_audit_v18a": (
        "9ff344ce001673f21a3782c813f2545f7503c19f8f0cc6be7d57ae64bfc8e7f3"
    ),
    "token_audit_tests_v18a": (
        "9c2be7bbc182bfceaa09cc9a68d168dc38889542437e2fd9ad1dd2b282dcf35e"
    ),
    "token_audit_artifact_v18a": (
        "df1d3810e988c3ece4ef921643ffe226fa7bb7f2f91edf2895865afe78c7ee6f"
    ),
    "prereg_module_v18a": (
        "683dace9ae2328dcf0d8c79e5a5b69ecdcc62dc55586f64376e4dc8fdcb4819f"
    ),
    "prereg_tests_v18a": (
        "36f0e5569834aa2f0f4f1dd4c83468f5857411ebcfffca5b3aaa17f9fc4cc68c"
    ),
    "preregistration_v18a": mechanics_v18a.PREREGISTRATION_FILE_SHA256_V18A,
}
FROZEN_PHASE_PATHS_V18A = {
    "frame_builder_v18a": Path(mechanics_v18a.frame_v18a.__file__).resolve(),
    "frame_tests_v18a": (ROOT / "test_build_eggroll_es_overlay_frame_v18a.py").resolve(),
    "frame_certificate_v18a": mechanics_v18a.frame_v18a.OUTPUT_PATH_V18A,
    "token_audit_v18a": Path(prereg_v18a.token_audit_v18a.__file__).resolve(),
    "token_audit_tests_v18a": (ROOT / "test_audit_eggroll_es_token_lengths_v18a.py").resolve(),
    "token_audit_artifact_v18a": prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A,
    "prereg_module_v18a": Path(prereg_v18a.__file__).resolve(),
    "prereg_tests_v18a": (
        ROOT / "test_eggroll_es_production_overlay_scan_preregistration_v18a.py"
    ).resolve(),
    "preregistration_v18a": prereg_v18a.OUTPUT_PATH_V18A,
}
IMPLEMENTATION_PATHS_V18A = {
    **driver_v13.IMPLEMENTATION_PATHS_V13,
    **FROZEN_PHASE_PATHS_V18A,
    **TRAINER_PHASE_PATHS_V18A,
    "runtime_driver_v18a": Path(__file__).resolve(),
    "runtime_tests_v18a": TEST_PATH_V18A,
}
FORBIDDEN_SURFACE_TOKENS_V18A = (
    "checkpoint", "update", "heldout", "validation", "ood", "benchmark",
    "eval", "save-model", "train-dataset",
)
FORBIDDEN_PERSISTED_KEYS_V18A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "unit_scores", "responses", "coefficients",
    "bootstrap_replicates", "bootstrap_draws", "row_content", "joint_ids",
    "row_sha256",
}
MOE_OVERRIDE_ENVIRONMENT_V18A = (
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)

canonical_sha256 = prereg_v18a.canonical_sha256
file_sha256 = prereg_v18a.file_sha256
anchor_v4 = anchor_v13.anchor_v4


def _without_self_v18a(value):
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _assert_train_only_argv_v18a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V18A):
            raise ValueError(f"v18a rejects forbidden runtime surface: {token}")


def _verify_commit_bindings_v18a():
    bindings = (
        (
            mechanics_v18a.frame_v18a.OUTPUT_PATH_V18A,
            BASE_FRAME_COMMIT_V18A,
            FROZEN_PHASE_HASHES_V18A["frame_certificate_v18a"],
        ),
        (
            Path(prereg_v18a.token_audit_v18a.__file__).resolve(),
            TOKEN_CORRECTION_COMMIT_V18A,
            FROZEN_PHASE_HASHES_V18A["token_audit_v18a"],
        ),
        (
            prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A,
            TOKEN_CORRECTION_COMMIT_V18A,
            FROZEN_PHASE_HASHES_V18A["token_audit_artifact_v18a"],
        ),
        (
            Path(prereg_v18a.__file__).resolve(),
            RUNTIME_CORRECTION_COMMIT_V18A,
            FROZEN_PHASE_HASHES_V18A["prereg_module_v18a"],
        ),
        (
            prereg_v18a.OUTPUT_PATH_V18A,
            RUNTIME_CORRECTION_COMMIT_V18A,
            FROZEN_PHASE_HASHES_V18A["preregistration_v18a"],
        ),
        (
            TRAINER_PATH_V18A,
            TRAINER_COMMIT_V18A,
            TRAINER_PHASE_HASHES_V18A["trainer_mechanics_v18a"],
        ),
        (
            TRAINER_TEST_PATH_V18A,
            TRAINER_COMMIT_V18A,
            TRAINER_PHASE_HASHES_V18A["trainer_tests_v18a"],
        ),
    )
    for path, commit, digest in bindings:
        mechanics_v18a._verify_commit_file_v18a(path, commit, digest)


def implementation_identity_v18a():
    inherited = driver_v13.implementation_identity_v13()
    _verify_commit_bindings_v18a()
    mechanics_v18a.load_hardened_preregistration_v18a()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V18A.items()
    }
    if {key: files[key] for key in inherited["files"]} != inherited["files"]:
        raise RuntimeError("v18a exact inherited V13 implementation changed")
    expected = {**FROZEN_PHASE_HASHES_V18A, **TRAINER_PHASE_HASHES_V18A}
    if {key: files[key]["file_sha256"] for key in expected} != expected:
        raise RuntimeError("v18a frozen offline or trainer phase changed")
    trainer_phase = {
        key: files[key] for key in TRAINER_PHASE_PATHS_V18A
    }
    return {
        "files": files,
        "inherited_v13_bundle_sha256": inherited["bundle_sha256"],
        "trainer_phase_bundle_sha256": canonical_sha256(trainer_phase),
        "bundle_sha256": canonical_sha256(files),
    }


def _source_provenance_v18a(implementation):
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
                f"v18a real launch requires committed source: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v18a source differs from HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    result = {
        "schema": "eggroll-es-committed-source-bundle-v18a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _parser_v18a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v18a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def _declared_moe_backend_v18a():
    return {
        "moe_backend": "default_triton",
        "override_environment": {
            name: None for name in MOE_OVERRIDE_ENVIRONMENT_V18A
        },
        "v16_task_ab_decision_retained": "default_triton",
    }


def _validate_moe_backend_environment_v18a():
    conflicts = {
        name: os.environ.get(name)
        for name in MOE_OVERRIDE_ENVIRONMENT_V18A
        if os.environ.get(name) not in (None, "")
    }
    if conflicts:
        raise ValueError("v18a requires every MoE backend override unset")
    return _declared_moe_backend_v18a()


def _load_bound_inputs_v18a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    plan = anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    preregistration = mechanics_v18a.load_hardened_preregistration_v18a()
    panel_bundle = mechanics_v18a.load_patch_panel_bundle_v18a()
    frozen = preregistration["frozen_recipe"]
    if (
        layer_bundle["plan_sha256"]
        != driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
        or Path(layer_bundle["path"]).resolve() != Path(plan["path"]).resolve()
        or layer_bundle["file_sha256"] != plan["file_sha256"]
        or layer_bundle["model_config_sha256"]
        != prereg_v18a.prereg_v17a.MODEL_CONFIG_SHA256_V17A
        or Path(frozen["model"]).resolve() != FROZEN_MODEL_V18A
        or frozen["layers"] != [20, 21, 22, 23]
        or frozen["alpha"] != 0.0
        or frozen["sigma"] != 0.0003
        or frozen["population_size"] != 32
        or frozen["layer_plan"]["plan_sha256"] != layer_bundle["plan_sha256"]
        or preregistration["scoring"]["dense_reward_config_sha256"]
        != anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
    ):
        raise RuntimeError("v18a model/layer/scoring preregistration changed")
    return preregistration, panel_bundle


def recipe_v18a(
    layer_bundle, preregistration, panel_bundle, implementation, moe_backend,
):
    if moe_backend != _declared_moe_backend_v18a():
        raise RuntimeError("v18a default Triton backend declaration changed")
    panels = {}
    for panel_name in mechanics_v18a.PANEL_NAMES_V18A:
        panel = panel_bundle["panels"][panel_name]
        panels[panel_name] = {
            "role": panel["role"],
            "arms": {
                arm: {
                    "rows": len(panel["arms"][arm]["questions"]),
                    "ordered_joint_identity_sha256": panel["arms"][arm][
                        "ordered_joint_identity_sha256"
                    ],
                    "ordered_row_identity_sha256": panel["arms"][arm][
                        "ordered_row_identity_sha256"
                    ],
                }
                for arm in mechanics_v18a.ARMS_V18A
            },
        }
    recipe = {
        "schema": "eggroll-es-production-patch-compat-recipe-v18a",
        "experiment_name": EXPERIMENT_NAME_V18A,
        "model": str(FROZEN_MODEL_V18A),
        "preregistration": copy.deepcopy(panel_bundle["preregistration"]),
        "frame": copy.deepcopy(panel_bundle["frame"]),
        "sources": copy.deepcopy(panel_bundle["sources"]),
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
            "basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
            "basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
            "seed_sha256": canonical_sha256(anchor_v13.PERTURBATION_SEEDS_V13),
            "signed_wave_schedule_sha256": canonical_sha256(
                mechanics_v18a.resident_signed_wave_schedule_v18a()
            ),
            "score_all_four_arms_before_restore": True,
            "restore_once_after_all_four_arms": True,
        },
        "scoring": {
            "objective": "per_example_mean_full_gold_answer_token_logprob",
            "dense_reward_config_sha256": (
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
            ),
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
            "requests_per_engine_per_arm_per_signed_wave": {
                arm: 5 * mechanics_v18a.frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm]
                for arm in mechanics_v18a.ARMS_V18A
            },
            "requests_per_engine_per_signed_wave_all_arms": 1070,
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
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V18A),
    }
    recipe["content_sha256_before_self_field"] = canonical_sha256(recipe)
    prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(recipe)
    return recipe


def validate_runtime_v18a(args, layer_bundle, implementation, recipe):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    if (
        not isinstance(args.v18a_dry_run, bool)
        or recipe.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v18a(recipe))
        or recipe.get("model_update_allowed") is not False
        or recipe.get("checkpoint_write_allowed") is not False
        or recipe.get("evaluation_surfaces_opened") is not False
        or recipe.get("moe_backend") != _declared_moe_backend_v18a()
        or recipe.get("hardware") != {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        }
        or recipe.get("scoring", {}).get(
            "requests_per_engine_per_signed_wave_all_arms"
        )
        != 1070
    ):
        raise ValueError("v18a frozen train-only runtime recipe changed")
    if not args.v18a_dry_run and (
        args.expected_implementation_bundle_sha256 is None
        or args.expected_recipe_sha256 is None
    ):
        raise ValueError("v18a real launch requires implementation and recipe hashes")
    if (
        args.expected_implementation_bundle_sha256 is not None
        and args.expected_implementation_bundle_sha256
        != implementation["bundle_sha256"]
    ):
        raise ValueError("v18a implementation bundle hash changed")
    if (
        args.expected_recipe_sha256 is not None
        and args.expected_recipe_sha256
        != recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v18a recipe hash changed")


def _request_identity_v18a(prompt_items):
    return canonical_sha256([item["prompt_token_ids"] for item in prompt_items])


def _score_panel_unit_outputs_v18a(dense_items, outputs, expected_rows):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = np.asarray([
        item["mean_answer_token_logprob"] for item in dense["examples"]
    ], dtype=np.float64)
    if rewards.shape != (expected_rows,) or not np.isfinite(rewards).all():
        raise RuntimeError("v18a per-unit dense score coverage changed")
    return rewards, canonical_sha256(dense)


class ProductionPatchCompatRuntimeMixinV18A:
    """One resident trainer with every update/evaluation surface closed."""

    def configure_production_patch_compat_v18a(
        self, panel_bundle, *, frozen_layer_plan,
    ):
        panel_bundle = mechanics_v18a.validate_patch_panel_bundle_v18a(
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
            raise ValueError("v18a requires one four-TP1-engine alpha-zero trainer")
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
        self._v4_reward_config_sha256 = anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v18a workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4", (4, "exact_reference"),
        )
        reference = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(reference, fresh=True)
        self._v18a_panel_bundle = panel_bundle
        return {
            "schema": "eggroll-es-production-patch-runtime-configuration-v18a",
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
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
        }

    def _prepared_fixed_batches_v18a(self):
        bundle = mechanics_v18a.validate_patch_panel_bundle_v18a(
            self._v18a_panel_bundle
        )
        prepared = {}
        identities = {}
        for arm in mechanics_v18a.ARMS_V18A:
            panels = {}
            prompt_items = []
            cursor = 0
            for panel_name in mechanics_v18a.PANEL_NAMES_V18A:
                batch = bundle["panels"][panel_name]["arms"][arm]
                prompts = [base.specialist_template(item) for item in batch["questions"]]
                dense_items = anchor_v4.prepare_gold_answer_items_v4(
                    self.tokenizer, prompts, batch["answers"],
                )
                current = [
                    {"prompt_token_ids": item["prompt_token_ids"]}
                    for item in dense_items
                ]
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(current)),
                    "templated_prompt_answer_sha256": canonical_sha256({
                        "prompts": prompts, "answers": batch["answers"],
                    }),
                }
                prompt_items.extend(current)
                cursor += len(current)
            expected = 5 * mechanics_v18a.frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm]
            if cursor != expected:
                raise RuntimeError("v18a fixed arm request count changed")
            prepared[arm] = {"panels": panels, "prompt_items": prompt_items}
            identities[arm] = _request_identity_v18a(prompt_items)
        self._v18a_fixed_request_identity = identities
        return prepared, identities

    def _assert_fixed_request_v18a(self, arm, prepared):
        if (
            arm not in mechanics_v18a.ARMS_V18A
            or _request_identity_v18a(prepared[arm]["prompt_items"])
            != self._v18a_fixed_request_identity[arm]
        ):
            raise RuntimeError("v18a fixed request identity changed")

    def _base_probe_v18a(self, prepared):
        result = {}
        for arm in mechanics_v18a.ARMS_V18A:
            self._assert_fixed_request_v18a(arm, prepared)
            outputs = anchor_v13.anchor_v11.anchor_v1.dispatch_eval_batch(
                self.engines,
                list(prepared[arm]["prompt_items"]),
                self._dense_sampling_params_v4(0),
                self._resolve,
            )
            panel_hashes = {}
            for panel_name in mechanics_v18a.PANEL_NAMES_V18A:
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                _rewards, dense_hash = _score_panel_unit_outputs_v18a(
                    panel["dense_items"], outputs[start:end], end - start,
                )
                panel_hashes[panel_name] = dense_hash
            result[arm] = canonical_sha256(panel_hashes)
        return result

    def _perturb_signed_wave_v18a(self, engine_seeds, negate):
        if len(engine_seeds) != 4:
            raise RuntimeError("v18a perturbation wave is partial")
        self._resolve([
            self.engines[index].collective_rpc.remote(
                "perturb_self_weights", args=(int(seed), 0.0003, bool(negate)),
            )
            for index, seed in enumerate(engine_seeds)
        ])

    def _score_resident_arm_v18a(
        self, arm, prepared, schedule_item, unit_scores, dense_commitments,
    ):
        self._assert_fixed_request_v18a(arm, prepared)
        prompts = prepared[arm]["prompt_items"]
        batches = self._resolve([
            engine.generate.remote(
                list(prompts), self._dense_sampling_params_v4(0), use_tqdm=False,
            )
            for engine in self.engines
        ])
        if (
            not isinstance(batches, list)
            or len(batches) != 4
            or any(len(batch) != len(prompts) for batch in batches)
        ):
            raise RuntimeError("v18a resident arm generation is incomplete")
        sign_index = mechanics_v18a.SIGNS_V18A.index(schedule_item["sign"])
        for engine_index, batch in enumerate(batches):
            direction_index = 4 * schedule_item["population_wave_index"] + engine_index
            for panel_index, panel_name in enumerate(
                mechanics_v18a.PANEL_NAMES_V18A
            ):
                panel = prepared[arm]["panels"][panel_name]
                start, end = panel["slice"]
                rewards, dense_hash = _score_panel_unit_outputs_v18a(
                    panel["dense_items"], batch[start:end], end - start,
                )
                unit_scores[arm][
                    panel_index, sign_index, direction_index,
                ] = rewards
                dense_commitments.append(dense_hash)
        return canonical_sha256(dense_commitments[-20:])

    def _restore_and_verify_signed_wave_v18a(self):
        self._restore_all_engines_exact()
        checks = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        if not anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks,
            lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v18a signed-wave exact reference restore failed")
        return canonical_sha256(checks)

    def _run_signed_wave_v18a(
        self, schedule_item, prepared, unit_scores, dense_commitments,
    ):
        restore_hashes = []
        captures = mechanics_v18a.execute_patch_resident_signed_wave_v18a(
            schedule_item,
            perturb=lambda seeds, negate: self._perturb_signed_wave_v18a(
                seeds, negate,
            ),
            score_arm=lambda arm: self._score_resident_arm_v18a(
                arm, prepared, schedule_item, unit_scores, dense_commitments,
            ),
            restore=lambda: restore_hashes.append(
                self._restore_and_verify_signed_wave_v18a()
            ),
        )
        if len(restore_hashes) != 1 or tuple(captures) != tuple(
            schedule_item["resident_arm_order"]
        ):
            raise RuntimeError("v18a signed-wave resident transaction changed")
        return restore_hashes[0]

    def estimate_production_patch_compat_v18a(self, seeds):
        seeds = [int(seed) for seed in seeds]
        schedule = mechanics_v18a.resident_signed_wave_schedule_v18a(seeds)
        prepared, request_identity = self._prepared_fixed_batches_v18a()
        pre_probe = self._base_probe_v18a(prepared)
        unit_scores = {
            arm: np.full(
                (
                    5,
                    2,
                    32,
                    mechanics_v18a.frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm],
                ),
                np.nan,
                dtype=np.float64,
            )
            for arm in mechanics_v18a.ARMS_V18A
        }
        dense_commitments = []
        restore_hashes = []
        for schedule_item in schedule:
            restore_hashes.append(self._run_signed_wave_v18a(
                schedule_item, prepared, unit_scores, dense_commitments,
            ))
        if (
            any(not np.isfinite(values).all() for values in unit_scores.values())
            or len(dense_commitments) != 4 * 5 * 2 * 32
            or len(restore_hashes) != 16
        ):
            raise RuntimeError("v18a patch population capture is incomplete")
        boundary = self._population_boundary_audit_v4(0)
        if (
            boundary.get("passed") is not True
            or boundary.get("audit_sha256")
            != canonical_sha256({
                key: value
                for key, value in boundary.items()
                if key != "audit_sha256"
            })
        ):
            raise RuntimeError("v18a population boundary audit failed")
        post_probe = self._base_probe_v18a(prepared)
        if pre_probe != post_probe:
            raise RuntimeError("v18a pre/post base probes drifted")
        compact = mechanics_v18a.build_compact_estimator_summary_v18a(
            unit_scores, self._v18a_panel_bundle,
        )
        unit_scores = None
        runtime_integrity = {
            "all_four_tp1_engines_every_signed_wave": True,
            "gpu_ids_zero_through_three_declared": True,
            "fixed_side_batch_identity_every_direction_sign_and_arm": True,
            "latin_arm_order_complete": True,
            "same_resident_perturbation_all_four_arms": True,
            "exact_reference_restored_once_per_signed_wave": True,
            "pre_post_base_probes_equal_all_four_arms": True,
            "population_boundary_audit_passed": True,
            "tokenizer_and_prompt_logprob_contract_passed": True,
            "all_integrity_audits_passed": True,
        }
        summary = {
            "schema": "eggroll-es-production-patch-compat-summary-v18a",
            "experiment_name": EXPERIMENT_NAME_V18A,
            "alpha": 0.0,
            "sigma": 0.0003,
            "model_update_applied": False,
            "evaluation_surfaces_opened": False,
            "frame_content_sha256": prereg_v18a.FLOW_CERTIFICATE_CONTENT_SHA256_V18A,
            "perturbation_basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
            "runtime_integrity": runtime_integrity,
            "arms": compact["arms"],
            "paired_bootstrap": compact["paired_bootstrap"],
            "persisted_response_vectors_or_row_content": False,
        }
        summary["content_sha256_before_self_field"] = canonical_sha256(summary)
        gate = mechanics_v18a.evaluate_patch_gate_v18a(summary)
        runtime_audit = {
            "schema": "eggroll-es-production-patch-runtime-compact-audit-v18a",
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
            "requests_per_engine_per_signed_wave": 1070,
            "per_unit_scores_persisted": False,
            "bootstrap_replicates_persisted": False,
        }
        runtime_audit["content_sha256_before_self_field"] = canonical_sha256(
            runtime_audit
        )
        return summary, gate, runtime_audit

    @staticmethod
    def _closed_surface_v18a(*_args, **_kwargs):
        raise RuntimeError("v18a closes every update checkpoint and eval entrypoint")

    configure_train_panels_v13 = _closed_surface_v18a
    estimate_train_panels_v13 = _closed_surface_v18a
    estimate_step_coefficients = _closed_surface_v18a
    apply_seed_coefficients = _closed_surface_v18a
    train_step = _closed_surface_v18a
    evaluate_handle = _closed_surface_v18a
    evaluate_population_on_batch = _closed_surface_v18a
    eval_step = _closed_surface_v18a
    fit = _closed_surface_v18a


def load_runtime_trainer_v18a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    parent = anchor_v13.load_trainer(layer_bundle)

    class ProductionPatchCompatRuntimeTrainerV18A(
        ProductionPatchCompatRuntimeMixinV18A, parent,
    ):
        pass

    return ProductionPatchCompatRuntimeTrainerV18A


def _make_trainer_v18a(layer_bundle):
    trainer_class = load_runtime_trainer_v18a(layer_bundle)
    return trainer_class(
        model_name=str(FROZEN_MODEL_V18A), checkpoint=None,
        sigma=0.0003, alpha=0.0, population_size=32,
        reward_shaping="z-scores", num_iterations=1, max_tokens=1,
        batch_size=275, mini_batch_size=275,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_V18A,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(FROZEN_OUTPUT_DIRECTORY_V18A),
    )


def _seal_v18a(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)


def _exclusive_write_json_v18a(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal_v18a(value)
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"v18a exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json_v18a(path, value):
    _seal_v18a(value)
    driver_v13.driver_v1.atomic_write_json(path, value)


def _new_launch_id_v18a():
    return f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"


def _claim_fresh_paths_v18a(attempt):
    root = (FROZEN_OUTPUT_DIRECTORY_V18A / EXPERIMENT_NAME_V18A).resolve()
    for _ in range(32):
        launch_id = _new_launch_id_v18a()
        attempt_path = root / "attempts" / f"{launch_id}.json"
        run_dir = root / "runs" / launch_id
        current = copy.deepcopy(attempt)
        current.update({
            "launch_id": launch_id,
            "run_directory": str(run_dir),
        })
        try:
            _exclusive_write_json_v18a(attempt_path, current)
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
            _rewrite_json_v18a(attempt_path, current)
            continue
        return attempt_path, run_dir, current
    raise RuntimeError("v18a could not claim collision-safe fresh launch paths")


def _recursive_keys_v18a(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _recursive_keys_v18a(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys_v18a(item)


def _validate_compact_configuration_v18a(configuration):
    if (
        not isinstance(configuration, dict)
        or configuration.get("schema")
        != "eggroll-es-production-patch-runtime-configuration-v18a"
        or configuration.get("panel_bundle_content_sha256")
        != mechanics_v18a.PANEL_BUNDLE_CONTENT_SHA256_V18A
        or configuration.get("worker_extension")
        != "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
        or configuration.get("engine_count") != 4
        or configuration.get("tp_per_engine") != 1
        or configuration.get("gpu_ids") != [0, 1, 2, 3]
        or configuration.get("model_update_allowed") is not False
        or configuration.get("checkpoint_write_allowed") is not False
        or configuration.get("evaluation_surfaces_opened") is not False
    ):
        raise RuntimeError("v18a compact configuration contract changed")
    return configuration


def _validate_compact_runtime_audit_v18a(audit):
    if (
        not isinstance(audit, dict)
        or audit.get("schema")
        != "eggroll-es-production-patch-runtime-compact-audit-v18a"
        or audit.get("signed_wave_count") != 16
        or audit.get("requests_per_engine_per_signed_wave") != 1070
        or audit.get("per_unit_scores_persisted") is not False
        or audit.get("bootstrap_replicates_persisted") is not False
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v18a(audit))
    ):
        raise RuntimeError("v18a compact runtime audit contract changed")
    return audit


def validate_compact_report_v18a(
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
        or report.get("schema")
        != "eggroll-es-production-patch-compat-report-v18a"
        or report.get("model_update_applied") is not False
        or report.get("checkpoint_written") is not False
        or report.get("evaluation_surfaces_opened") is not False
        or report.get("recipe") != expected_recipe
        or report.get("implementation") != expected_implementation
        or report.get("recipe", {}).get("implementation_bundle_sha256")
        != report.get("implementation", {}).get("bundle_sha256")
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v18a(report))
        or report.get("summary", {}).get(
            "persisted_response_vectors_or_row_content"
        )
        is not False
        or report.get("gate")
        != mechanics_v18a.evaluate_patch_gate_v18a(report.get("summary"))
        or any(
            str(key).lower() in FORBIDDEN_PERSISTED_KEYS_V18A
            for key in _recursive_keys_v18a(report)
        )
    ):
        raise RuntimeError("v18a compact report contract changed")
    _validate_compact_configuration_v18a(report["configuration"])
    _validate_compact_runtime_audit_v18a(report["runtime_audit"])
    prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(report)
    return report


def run_exact_v18a(layer_bundle, panel_bundle, implementation, recipe):
    if _validate_moe_backend_environment_v18a() != recipe.get("moe_backend"):
        raise ValueError("v18a runtime MoE backend differs from sealed recipe")
    provenance = _source_provenance_v18a(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v18a",
        "status": "launching",
        "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V18A,
        "recipe": recipe,
        "source_provenance": provenance,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
    }
    attempt_path, run_dir, attempt = _claim_fresh_paths_v18a(attempt)
    report_path = run_dir / REPORT_NAME_V18A
    trainer = None
    failure = None
    configuration = summary = gate = runtime_audit = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v18a(layer_bundle)
        configuration = trainer.configure_production_patch_compat_v18a(
            panel_bundle, frozen_layer_plan=layer_bundle,
        )
        summary, gate, runtime_audit = (
            trainer.estimate_production_patch_compat_v18a(
                anchor_v13.PERTURBATION_SEEDS_V13
            )
        )
        if gate != mechanics_v18a.evaluate_patch_gate_v18a(summary):
            raise RuntimeError("v18a hardened gate recomputation changed")
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
            "phase": "inside_patch_train_only_runtime",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
        })
        _rewrite_json_v18a(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-production-patch-compat-report-v18a",
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
    validate_compact_report_v18a(
        report,
        expected_recipe=recipe,
        expected_implementation=implementation,
    )
    try:
        _exclusive_write_json_v18a(report_path, report)
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
        _rewrite_json_v18a(attempt_path, attempt)
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
    _rewrite_json_v18a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v18a(argv)
    moe_backend = _validate_moe_backend_environment_v18a()
    layer_bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v18a().parse_args(remaining)
    implementation = implementation_identity_v18a()
    preregistration, panel_bundle = _load_bound_inputs_v18a(layer_bundle)
    recipe = recipe_v18a(
        layer_bundle, preregistration, panel_bundle, implementation, moe_backend,
    )
    validate_runtime_v18a(args, layer_bundle, implementation, recipe)
    if args.v18a_dry_run:
        payload = {
            "schema": "eggroll-es-production-patch-compat-dry-run-v18a",
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
        prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v18a(layer_bundle, panel_bundle, implementation, recipe)


if __name__ == "__main__":
    main()
