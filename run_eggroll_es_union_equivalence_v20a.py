#!/usr/bin/env python3
"""Fail-closed raw-versus-union scoring equivalence gate for V20A."""

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

import eggroll_es_union_request_plan_v20a as union_v20a
import run_eggroll_es_disjoint_tier_attribution_v19a as driver_v19a
import train_eggroll_es_nested_tier_interaction_v20a as mechanics_v20a


ROOT = Path(__file__).resolve().parent
prereg_v20a = mechanics_v20a.prereg_v20a
frame_v20a = mechanics_v20a.frame_v20a
anchor_v13 = driver_v19a.anchor_v13
anchor_v11 = driver_v19a.anchor_v11
anchor_v4 = driver_v19a.anchor_v4
base = driver_v19a.base

FROZEN_MODEL_V20A = driver_v19a.FROZEN_MODEL_V19A
FROZEN_OUTPUT_DIRECTORY_V20A = driver_v19a.FROZEN_OUTPUT_DIRECTORY_V19A
EXPERIMENT_NAME_V20A = prereg_v20a.EXPERIMENT_NAME_V20A + "_union_equivalence_gate"
REPORT_NAME_V20A = "union_scoring_equivalence_v20a.json"
TEST_PATH_V20A = (ROOT / "test_run_eggroll_es_union_equivalence_v20a.py").resolve()

SEALED_FOUNDATION_COMMIT_V20A = "f8860e14c693020badf25985cb2ba6b4d4339e30"
SEALED_MECHANICS_COMMIT_V20A = "d7f783d264a13f00a6813fe11490621da629df68"
SEALED_UNION_COMMIT_V20A = "cc342be6374b1cb57479119c0789ad67f19467e0"

FOUNDATION_PATHS_V20A = {
    "frame_builder_v20a": Path(frame_v20a.__file__).resolve(),
    "frame_certificate_v20a": frame_v20a.OUTPUT_PATH_V20A,
    "prereg_module_v20a": Path(prereg_v20a.__file__).resolve(),
    "preregistration_v20a": prereg_v20a.OUTPUT_PATH_V20A,
}
FOUNDATION_HASHES_V20A = {
    "frame_builder_v20a": mechanics_v20a.FRAME_BUILDER_FILE_SHA256_V20A,
    "frame_certificate_v20a": mechanics_v20a.FRAME_CERTIFICATE_FILE_SHA256_V20A,
    "prereg_module_v20a": mechanics_v20a.PREREGISTRATION_BUILDER_FILE_SHA256_V20A,
    "preregistration_v20a": mechanics_v20a.PREREGISTRATION_FILE_SHA256_V20A,
}
MECHANICS_PATHS_V20A = {
    "draw_plan_builder_v20a": Path(mechanics_v20a.draw_v20a.__file__).resolve(),
    "draw_plan_v20a": mechanics_v20a.draw_v20a.OUTPUT_PATH_V20A,
    "draw_plan_tests_v20a": (
        ROOT / "test_build_eggroll_es_nested_tier_draw_plan_v20a.py"
    ).resolve(),
    "trainer_mechanics_v20a": Path(mechanics_v20a.__file__).resolve(),
    "trainer_tests_v20a": (
        ROOT / "test_train_eggroll_es_nested_tier_interaction_v20a.py"
    ).resolve(),
}
MECHANICS_HASHES_V20A = {
    "draw_plan_builder_v20a": mechanics_v20a.DRAW_PLAN_BUILDER_FILE_SHA256_V20A,
    "draw_plan_v20a": mechanics_v20a.DRAW_PLAN_FILE_SHA256_V20A,
    "draw_plan_tests_v20a": (
        "6b2397e0b7a53d725de03525b228bc3837fe904da228ff6fe16171965ed7479e"
    ),
    "trainer_mechanics_v20a": (
        "52774f35de92421772c86ee53d79d2f8b9db7e21e8d4690cdf3fd1163eabee34"
    ),
    "trainer_tests_v20a": (
        "e7282fd7ed48eb54d3c593e4dea49b27fb59bb2620510dd4ccf8d579bdacd831"
    ),
}
UNION_PATHS_V20A = {
    "union_planner_v20a": Path(union_v20a.__file__).resolve(),
    "union_planner_tests_v20a": (
        ROOT / "test_eggroll_es_union_request_plan_v20a.py"
    ).resolve(),
}
UNION_HASHES_V20A = {
    "union_planner_v20a": prereg_v20a.UNION_PLANNER_FILE_SHA256_V20A,
    "union_planner_tests_v20a": prereg_v20a.UNION_PLANNER_TEST_FILE_SHA256_V20A,
}

FORBIDDEN_SURFACE_TOKENS_V20A = (
    "checkpoint", "update", "heldout", "validation", "ood", "benchmark",
    "eval", "save-model", "train-dataset", "promote", "attribution-run",
)
FORBIDDEN_PERSISTED_KEYS_V20A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "tokens", "row_sha256",
    "ordered_row_identity_sha256", "joint_ids", "ordered_joint_identity_sha256",
    "unit_scores", "responses", "coefficients", "bootstrap_replicates",
    "bootstrap_draws", "row_content", "union_prompt_items",
    "arm_panel_union_indices",
}
MOE_OVERRIDE_ENVIRONMENT_V20A = driver_v19a.MOE_OVERRIDE_ENVIRONMENT_V19A

canonical_sha256 = prereg_v20a.canonical_sha256
file_sha256 = prereg_v20a.file_sha256


def _without_self_v20a(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _assert_train_only_argv_v20a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V20A):
            raise ValueError(f"v20a rejects forbidden runtime surface: {token}")


def _verify_commit_file_v20a(path, commit, digest):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v20a committed artifact changed: {relative}")


def _verify_commit_bindings_v20a():
    for paths, hashes, commit in (
        (FOUNDATION_PATHS_V20A, FOUNDATION_HASHES_V20A, SEALED_FOUNDATION_COMMIT_V20A),
        (MECHANICS_PATHS_V20A, MECHANICS_HASHES_V20A, SEALED_MECHANICS_COMMIT_V20A),
        (UNION_PATHS_V20A, UNION_HASHES_V20A, SEALED_UNION_COMMIT_V20A),
    ):
        for key, path in paths.items():
            _verify_commit_file_v20a(path, commit, hashes[key])


def implementation_identity_v20a():
    inherited = driver_v19a.implementation_identity_v19a()
    _verify_commit_bindings_v20a()
    paths = {
        **FOUNDATION_PATHS_V20A,
        **MECHANICS_PATHS_V20A,
        **UNION_PATHS_V20A,
        "runtime_driver_v20a": Path(__file__).resolve(),
        "runtime_tests_v20a": TEST_PATH_V20A,
    }
    files = {
        key: {"path": str(path), "file_sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    phases = {
        "foundation_bundle_sha256": canonical_sha256({
            key: files[key] for key in FOUNDATION_PATHS_V20A
        }),
        "mechanics_bundle_sha256": canonical_sha256({
            key: files[key] for key in MECHANICS_PATHS_V20A
        }),
        "union_planner_bundle_sha256": canonical_sha256({
            key: files[key] for key in UNION_PATHS_V20A
        }),
    }
    return {
        "files": files,
        "inherited_v19a_bundle_sha256": inherited["bundle_sha256"],
        **phases,
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
        try:
            raw = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"v20a real launch requires committed source: {relative}") from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v20a source differs from HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    value = {
        "schema": "eggroll-es-committed-source-bundle-v20a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _declared_moe_backend_v20a():
    return {
        "moe_backend": "default_triton",
        "override_environment": {
            name: None for name in MOE_OVERRIDE_ENVIRONMENT_V20A
        },
        "v16_task_ab_decision_retained": "default_triton",
    }


def _validate_moe_backend_environment_v20a():
    conflicts = {
        name: os.environ.get(name)
        for name in MOE_OVERRIDE_ENVIRONMENT_V20A
        if os.environ.get(name) not in (None, "")
    }
    if conflicts:
        raise ValueError("v20a requires every MoE backend override unset")
    return _declared_moe_backend_v20a()


def equivalence_signed_wave_v20a():
    seeds = list(prereg_v20a.PERTURBATION_SEEDS_V20A[:4])
    return {
        "signed_wave_index": 0,
        "population_wave_index": 0,
        "sign": "plus",
        "negate": False,
        "engine_seeds": seeds,
        "basis_sha256": prereg_v20a.PERTURBATION_BASIS_SHA256_V20A,
        "restore_exact_reference_after_wave": True,
    }


def _load_bound_inputs_v20a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    preregistration = mechanics_v20a.load_hardened_preregistration_v20a()
    panel_bundle = mechanics_v20a.load_panel_bundle_v20a()
    frozen = preregistration["frozen_recipe"]
    if (
        Path(frozen["model"]).resolve() != FROZEN_MODEL_V20A
        or frozen["layers"] != [20, 21, 22, 23]
        or frozen["alpha"] != 0.0
        or frozen["sigma"] != 0.0003
        or frozen["population_size"] != 32
        or frozen["layer_plan"]["plan_sha256"] != layer_bundle["plan_sha256"]
        or frozen["layer_plan"]["file_sha256"] != layer_bundle["file_sha256"]
        or frozen["layer_plan"]["model_config_sha256"]
        != layer_bundle["model_config_sha256"]
        or frozen["perturbation_basis_sha256"]
        != prereg_v20a.PERTURBATION_BASIS_SHA256_V20A
        or panel_bundle["content_sha256_before_self_field"]
        != mechanics_v20a.PANEL_BUNDLE_CONTENT_SHA256_V20A
    ):
        raise RuntimeError("v20a model layer basis or panel binding changed")
    return preregistration, panel_bundle


def recipe_v20a(layer_bundle, preregistration, panel_bundle, implementation, moe):
    if moe != _declared_moe_backend_v20a():
        raise RuntimeError("v20a default Triton declaration changed")
    value = {
        "schema": "eggroll-es-union-scoring-equivalence-recipe-v20a",
        "experiment_name": EXPERIMENT_NAME_V20A,
        "model": str(FROZEN_MODEL_V20A),
        "layer_plan": copy.deepcopy(preregistration["frozen_recipe"]["layer_plan"]),
        "layers": [20, 21, 22, 23],
        "sigma": 0.0003,
        "alpha": 0.0,
        "population_size": 32,
        "perturbation_basis_sha256": prereg_v20a.PERTURBATION_BASIS_SHA256_V20A,
        "equivalence_signed_wave": equivalence_signed_wave_v20a(),
        "reference_state_equivalence_required": True,
        "perturbed_state_equivalence_required": True,
        "raw_arm_scoring": {
            "authoritative": True,
            "requests_per_engine": 1020,
            "requests_by_arm": {
                "production_only": 240,
                "patch_tier_2_only": 250,
                "patch_tiers_2_3": 260,
                "patch_all_tiers": 270,
            },
        },
        "union_scoring": {
            "planner_file_sha256": prereg_v20a.UNION_PLANNER_FILE_SHA256_V20A,
            "deduplicate_exact_token_sequences_only": True,
            "exact_unique_count_selected_at_launch": False,
            "bit_exact_per_arm_panel_scores_required": True,
            "bit_exact_dense_commitments_required": True,
        },
        "panel_bundle_content_sha256": panel_bundle[
            "content_sha256_before_self_field"
        ],
        "frame_content_sha256": mechanics_v20a.FRAME_CERTIFICATE_CONTENT_SHA256_V20A,
        "preregistration_content_sha256": (
            mechanics_v20a.PREREGISTRATION_CONTENT_SHA256_V20A
        ),
        "draw_plan_content_sha256": mechanics_v20a.DRAW_PLAN_CONTENT_SHA256_V20A,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "mechanics_bundle_sha256": implementation["mechanics_bundle_sha256"],
        "hardware": {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_both_equivalence_states": True,
        },
        "moe_backend": copy.deepcopy(moe),
        "authority": {
            "may_authorize_union_scoring_for_later_v20a_train_only_attribution": True,
            "may_launch_v20a_attribution": False,
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
    parser.add_argument("--v20a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_v20a(args, layer_bundle, implementation, recipe):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    if (
        recipe.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v20a(recipe))
        or recipe.get("alpha") != 0.0
        or recipe.get("hardware")
        != {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_both_equivalence_states": True,
        }
        or recipe.get("authority", {}).get("may_launch_v20a_attribution") is not False
        or recipe.get("authority", {}).get("model_update_allowed") is not False
        or recipe.get("authority", {}).get("checkpoint_write_allowed") is not False
        or recipe.get("authority", {}).get("evaluation_allowed") is not False
        or recipe.get("authority", {}).get("dataset_promotion_allowed") is not False
    ):
        raise ValueError("v20a equivalence recipe changed")
    if not args.v20a_dry_run and (
        args.expected_implementation_bundle_sha256 is None
        or args.expected_recipe_sha256 is None
    ):
        raise ValueError("v20a real launch requires implementation and recipe hashes")
    if (
        args.expected_implementation_bundle_sha256 is not None
        and args.expected_implementation_bundle_sha256 != implementation["bundle_sha256"]
    ):
        raise ValueError("v20a implementation bundle hash changed")
    if (
        args.expected_recipe_sha256 is not None
        and args.expected_recipe_sha256 != recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v20a recipe hash changed")


def _request_identity_v20a(items):
    return canonical_sha256([item["prompt_token_ids_sha256"] for item in items])


def compare_dense_equivalence_v20a(raw, union):
    """Return only aggregate equality commitments; never persist scores."""
    if set(raw) != set(union):
        raise RuntimeError("v20a raw/union dense coverage changed")
    raw_commitments = []
    union_commitments = []
    score_commitments = []
    for key in sorted(raw):
        raw_scores, raw_hash = raw[key]
        union_scores, union_hash = union[key]
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        union_scores = np.asarray(union_scores, dtype=np.float64)
        if (
            raw_hash != union_hash
            or raw_scores.shape != union_scores.shape
            or not np.array_equal(raw_scores, union_scores)
            or not np.isfinite(raw_scores).all()
        ):
            raise RuntimeError("v20a raw/union scoring is not bit-exact")
        raw_commitments.append(raw_hash)
        union_commitments.append(union_hash)
        score_commitments.append(hashlib.sha256(raw_scores.tobytes()).hexdigest())
    return {
        "entry_count": len(raw),
        "raw_dense_commitments_sha256": canonical_sha256(raw_commitments),
        "union_dense_commitments_sha256": canonical_sha256(union_commitments),
        "per_unit_score_bytes_sha256": canonical_sha256(score_commitments),
        "all_per_arm_panel_scores_bit_exact": True,
        "all_dense_commitments_bit_exact": True,
        "scores_or_outputs_persisted": False,
    }


class UnionScoringEquivalenceRuntimeMixinV20A(
    driver_v19a.DisjointTierAttributionRuntimeMixinV19A
):
    def configure_union_equivalence_v20a(self, panel_bundle, *, frozen_layer_plan):
        panel_bundle = mechanics_v20a.validate_panel_bundle_v20a(
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
            raise ValueError("v20a requires one four-TP1 alpha-zero trainer")
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
            raise RuntimeError("v20a workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4", (4, "exact_reference")
        )
        reference = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(reference, fresh=True)
        self._v20a_panel_bundle = panel_bundle
        return {
            "schema": "eggroll-es-union-equivalence-runtime-configuration-v20a",
            "layer_plan_install_sha256": canonical_sha256(install),
            "reference_identity_sha256": canonical_sha256(reference["reference_identity"]),
            "unselected_origin_sha256": install["unselected_origin_sha256"],
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_allowed": False,
            "attribution_runtime_opened": False,
        }

    def _prepared_fixed_batches_v20a(self):
        bundle = mechanics_v20a.validate_panel_bundle_v20a(self._v20a_panel_bundle)
        prepared = {}
        boundaries = []
        for arm in mechanics_v20a.ARMS_V20A:
            panels = {}
            flat = []
            cursor = 0
            for panel_name in mechanics_v20a.PANEL_NAMES_V20A:
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
                        raise RuntimeError("v20a fixed token boundary changed")
                    boundaries.append({
                        "arm": arm,
                        "panel": panel_name,
                        "identity_sha256": item["prompt_token_ids_sha256"],
                        "prompt_token_count": item["prompt_token_count"],
                        "answer_token_start": item["answer_token_start"],
                        "answer_token_count": item["answer_token_count"],
                    })
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(dense_items)),
                }
                flat.extend(dense_items)
                cursor += len(dense_items)
            expected = 10 * frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]
            if cursor != expected:
                raise RuntimeError("v20a fixed arm request count changed")
            prepared[arm] = {"panels": panels, "prompt_items": flat}
        identities = {
            arm: _request_identity_v20a(value["prompt_items"])
            for arm, value in prepared.items()
        }
        self._v20a_fixed_request_identity = identities
        self._v20a_token_boundary_audit_sha256 = canonical_sha256({
            "schema": "eggroll-es-fixed-token-boundary-audit-v20a",
            "frozen_total_token_cap": 1024,
            "records": boundaries,
        })
        return prepared, identities

    def _generate_raw_v20a(self, prepared):
        result = {engine_index: {} for engine_index in range(4)}
        for arm in mechanics_v20a.ARMS_V20A:
            items = prepared[arm]["prompt_items"]
            prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in items]
            batches = self._resolve([
                engine.generate.remote(
                    list(prompts), self._dense_sampling_params_v4(0), use_tqdm=False
                )
                for engine in self.engines
            ])
            if len(batches) != 4 or any(len(batch) != len(prompts) for batch in batches):
                raise RuntimeError("v20a raw generation coverage changed")
            for engine_index, outputs in enumerate(batches):
                for panel in mechanics_v20a.PANEL_NAMES_V20A:
                    contract = prepared[arm]["panels"][panel]
                    start, end = contract["slice"]
                    result[engine_index][(arm, panel)] = (
                        contract["dense_items"], outputs[start:end]
                    )
        return result

    def _generate_union_v20a(self, prepared, union_runtime):
        union_items = union_runtime["union_prompt_items"]
        prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in union_items]
        batches = self._resolve([
            engine.generate.remote(
                list(prompts), self._dense_sampling_params_v4(0), use_tqdm=False
            )
            for engine in self.engines
        ])
        if len(batches) != 4 or any(len(batch) != len(prompts) for batch in batches):
            raise RuntimeError("v20a union generation coverage changed")
        result = {engine_index: {} for engine_index in range(4)}
        for engine_index, outputs in enumerate(batches):
            for arm in mechanics_v20a.ARMS_V20A:
                rebuilt = union_v20a.reconstruct_arm_outputs_v20a(
                    union_runtime, arm, mechanics_v20a.PANEL_NAMES_V20A, outputs
                )
                for panel in mechanics_v20a.PANEL_NAMES_V20A:
                    result[engine_index][(arm, panel)] = (
                        prepared[arm]["panels"][panel]["dense_items"],
                        rebuilt[panel],
                    )
        return result

    @staticmethod
    def _score_generated_v20a(generated):
        result = {}
        for engine_index, entries in generated.items():
            for (arm, panel), (dense_items, outputs) in entries.items():
                result[(engine_index, arm, panel)] = (
                    driver_v19a._score_panel_unit_outputs_v19a(
                        dense_items, outputs, len(dense_items)
                    )
                )
        return result

    def _equivalence_state_v20a(self, state, prepared, union_runtime, union_audit):
        raw = self._score_generated_v20a(self._generate_raw_v20a(prepared))
        union = self._score_generated_v20a(
            self._generate_union_v20a(prepared, union_runtime)
        )
        equality = compare_dense_equivalence_v20a(raw, union)
        raw = union = None
        return {
            "state": state,
            "raw_requests_per_engine": 1020,
            "unique_union_requests_per_engine": union_audit["unique_request_count"],
            "eliminated_duplicate_requests_per_engine": union_audit[
                "eliminated_duplicate_request_count"
            ],
            "raw_to_unique_ratio": union_audit["raw_to_unique_ratio"],
            "union_audit_content_sha256": union_audit[
                "content_sha256_before_self_field"
            ],
            **equality,
        }

    def run_union_equivalence_v20a(self):
        prepared, identities = self._prepared_fixed_batches_v20a()
        union_runtime, union_audit = union_v20a.build_union_request_plan_v20a(
            prepared, mechanics_v20a.ARMS_V20A, mechanics_v20a.PANEL_NAMES_V20A
        )
        if (
            union_audit["raw_request_count"] != 1020
            or union_audit["unique_request_count"] >= 1020
            or union_audit["contains_token_ids_or_row_content"] is not False
        ):
            raise RuntimeError("v20a union plan aggregate contract changed")
        reference = self._equivalence_state_v20a(
            "exact_reference", prepared, union_runtime, union_audit
        )
        wave = equivalence_signed_wave_v20a()
        perturbed = None
        restore_hash = None
        try:
            self._perturb_signed_wave_v19a(wave["engine_seeds"], wave["negate"])
            perturbed = self._equivalence_state_v20a(
                "preregistered_perturbed_plus_wave_0",
                prepared,
                union_runtime,
                union_audit,
            )
        finally:
            restore_hash = self._restore_and_verify_signed_wave_v19a()
        if perturbed is None or restore_hash is None:
            raise RuntimeError("v20a perturbed equivalence transaction incomplete")
        boundary = self._population_boundary_audit_v4(0)
        unselected, unselected_sha = driver_v19a._validate_population_boundary_v19a(
            boundary, self._v4_layer_plan_install
        )
        union_runtime = prepared = None
        summary = {
            "schema": "eggroll-es-union-equivalence-summary-v20a",
            "experiment_name": EXPERIMENT_NAME_V20A,
            "reference_equivalence": reference,
            "perturbed_equivalence": perturbed,
            "exact_reference_restored_after_perturbed_wave": True,
            "all_four_tp1_engines_both_states": True,
            "raw_arm_scoring_authoritative": True,
            "union_scoring_authorized_for_later_v20a_train_only_attribution": True,
            "v20a_attribution_run_authorized": False,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_applied": False,
            "scores_outputs_tokens_or_row_content_persisted": False,
        }
        summary["content_sha256_before_self_field"] = canonical_sha256(summary)
        gate = evaluate_equivalence_gate_v20a(summary)
        audit = {
            "schema": "eggroll-es-union-equivalence-runtime-audit-v20a",
            "fixed_request_identity_sha256": canonical_sha256(identities),
            "token_boundary_audit_sha256": self._v20a_token_boundary_audit_sha256,
            "union_plan_content_sha256": union_audit["content_sha256_before_self_field"],
            "restore_check_sha256": restore_hash,
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "unselected_origin_sha256": boundary["unselected_origin_sha256"],
            "unselected_origin_audit_sha256": unselected_sha,
            "unselected_origin_audit_passed": unselected["passed"],
            "equivalence_state_count": 2,
            "engine_count": 4,
            "dense_comparisons_per_state": 160,
            "raw_requests_per_engine_per_state": 1020,
            "unique_union_requests_per_engine_per_state": union_audit[
                "unique_request_count"
            ],
            "tokens_scores_outputs_or_row_content_persisted": False,
        }
        audit["content_sha256_before_self_field"] = canonical_sha256(audit)
        return summary, gate, audit

    @staticmethod
    def _closed_surface_v20a(*_args, **_kwargs):
        raise RuntimeError(
            "v20a equivalence closes attribution update checkpoint evaluation and promotion"
        )

    configure_disjoint_tier_attribution_v19a = _closed_surface_v20a
    estimate_disjoint_tier_attribution_v19a = _closed_surface_v20a
    build_compact_estimator_summary_v20a = _closed_surface_v20a
    train_step = _closed_surface_v20a
    fit = _closed_surface_v20a
    eval_step = _closed_surface_v20a
    evaluate_handle = _closed_surface_v20a
    evaluate_population_on_batch = _closed_surface_v20a
    apply_seed_coefficients = _closed_surface_v20a


def evaluate_equivalence_gate_v20a(summary):
    reference = summary.get("reference_equivalence", {})
    perturbed = summary.get("perturbed_equivalence", {})
    passed = all(
        state.get("all_per_arm_panel_scores_bit_exact") is True
        and state.get("all_dense_commitments_bit_exact") is True
        and state.get("entry_count") == 160
        and state.get("raw_dense_commitments_sha256")
        == state.get("union_dense_commitments_sha256")
        for state in (reference, perturbed)
    ) and (
        summary.get("exact_reference_restored_after_perturbed_wave") is True
        and summary.get("all_four_tp1_engines_both_states") is True
        and summary.get("raw_arm_scoring_authoritative") is True
        and summary.get(
            "union_scoring_authorized_for_later_v20a_train_only_attribution"
        ) is True
        and summary.get("v20a_attribution_run_authorized") is False
        and summary.get("model_update_applied") is False
        and summary.get("checkpoint_written") is False
        and summary.get("evaluation_surfaces_opened") is False
        and summary.get("dataset_promotion_applied") is False
        and summary.get("scores_outputs_tokens_or_row_content_persisted") is False
    )
    return {
        "schema": "eggroll-es-union-equivalence-gate-v20a",
        "reference_bit_exact": bool(passed and reference.get(
            "all_per_arm_panel_scores_bit_exact"
        )),
        "perturbed_bit_exact": bool(passed and perturbed.get(
            "all_per_arm_panel_scores_bit_exact"
        )),
        "union_scoring_authorized_for_later_v20a_train_only_attribution": bool(passed),
        "v20a_attribution_run_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
        "decision": (
            "authorize_union_scoring_implementation_only_for_later_separately_launched_v20a_train_only_attribution"
            if passed else "retain_authoritative_raw_arm_scoring"
        ),
    }


def load_runtime_trainer_v20a(layer_bundle):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_bundle)
    parent = anchor_v13.load_trainer(layer_bundle)

    class UnionScoringEquivalenceRuntimeTrainerV20A(
        UnionScoringEquivalenceRuntimeMixinV20A, parent
    ):
        pass

    return UnionScoringEquivalenceRuntimeTrainerV20A


def _make_trainer_v20a(layer_bundle):
    trainer_class = load_runtime_trainer_v20a(layer_bundle)
    return trainer_class(
        model_name=str(FROZEN_MODEL_V20A),
        checkpoint=None,
        sigma=0.0003,
        alpha=0.0,
        population_size=32,
        reward_shaping="z-scores",
        num_iterations=1,
        max_tokens=1,
        batch_size=270,
        mini_batch_size=270,
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
        experiment_name=EXPERIMENT_NAME_V20A,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(FROZEN_OUTPUT_DIRECTORY_V20A),
    )


def _seal_v20a(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)


def _exclusive_write_json_v20a(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal_v20a(value)
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"v20a exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json_v20a(path, value):
    _seal_v20a(value)
    driver_v19a.driver_v13.driver_v1.atomic_write_json(path, value)


def _claim_fresh_paths_v20a(attempt):
    root = (FROZEN_OUTPUT_DIRECTORY_V20A / EXPERIMENT_NAME_V20A).resolve()
    for _ in range(32):
        launch_id = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"
        attempt_path = root / "attempts" / f"{launch_id}.json"
        run_dir = root / "runs" / launch_id
        current = copy.deepcopy(attempt)
        current.update({"launch_id": launch_id, "run_directory": str(run_dir)})
        try:
            _exclusive_write_json_v20a(attempt_path, current)
        except FileExistsError:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        return attempt_path, run_dir, current
    raise RuntimeError("v20a could not claim fresh launch paths")


def _recursive_keys_v20a(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _recursive_keys_v20a(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys_v20a(item)


def validate_compact_report_v20a(report, *, expected_recipe, expected_implementation):
    configuration = report.get("configuration", {}) if isinstance(report, dict) else {}
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    runtime_audit = report.get("runtime_audit", {}) if isinstance(report, dict) else {}
    if (
        not isinstance(report, dict)
        or set(report) != {
            "schema", "recipe", "configuration", "runtime_audit", "summary",
            "gate", "implementation", "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
            "attribution_run_opened", "content_sha256_before_self_field",
        }
        or report.get("schema") != "eggroll-es-union-equivalence-report-v20a"
        or report.get("recipe") != expected_recipe
        or report.get("implementation") != expected_implementation
        or report.get("recipe", {}).get("implementation_bundle_sha256")
        != report.get("implementation", {}).get("bundle_sha256")
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v20a(report))
        or summary.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v20a(summary))
        or runtime_audit.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v20a(runtime_audit))
        or configuration.get("engine_count") != 4
        or configuration.get("tp_per_engine") != 1
        or configuration.get("gpu_ids") != [0, 1, 2, 3]
        or configuration.get("model_update_allowed") is not False
        or configuration.get("checkpoint_write_allowed") is not False
        or configuration.get("evaluation_surfaces_opened") is not False
        or configuration.get("dataset_promotion_allowed") is not False
        or configuration.get("attribution_runtime_opened") is not False
        or runtime_audit.get("equivalence_state_count") != 2
        or runtime_audit.get("engine_count") != 4
        or runtime_audit.get("dense_comparisons_per_state") != 160
        or runtime_audit.get("raw_requests_per_engine_per_state") != 1020
        or runtime_audit.get(
            "tokens_scores_outputs_or_row_content_persisted"
        ) is not False
        or report.get("model_update_applied") is not False
        or report.get("checkpoint_written") is not False
        or report.get("evaluation_surfaces_opened") is not False
        or report.get("dataset_promotion_applied") is not False
        or report.get("attribution_run_opened") is not False
        or report.get("gate") != evaluate_equivalence_gate_v20a(report.get("summary", {}))
        or report.get("gate", {}).get("v20a_attribution_run_authorized") is not False
        or report.get("gate", {}).get("model_update_authorized") is not False
        or report.get("gate", {}).get("evaluation_authorized") is not False
        or any(
            str(key).lower() in FORBIDDEN_PERSISTED_KEYS_V20A
            for key in _recursive_keys_v20a(report)
        )
    ):
        raise RuntimeError("v20a compact equivalence report changed")
    return report


def run_exact_v20a(layer_bundle, panel_bundle, implementation, recipe):
    provenance = _source_provenance_v20a(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v20a",
        "status": "launching",
        "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V20A,
        "recipe": recipe,
        "source_provenance": provenance,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
        "attribution_run_opened": False,
    }
    attempt_path, run_dir, attempt = _claim_fresh_paths_v20a(attempt)
    report_path = run_dir / REPORT_NAME_V20A
    trainer = None
    failure = None
    configuration = summary = gate = audit = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v20a(layer_bundle)
        configuration = trainer.configure_union_equivalence_v20a(
            panel_bundle, frozen_layer_plan=layer_bundle
        )
        summary, gate, audit = trainer.run_union_equivalence_v20a()
        if gate != evaluate_equivalence_gate_v20a(summary):
            raise RuntimeError("v20a equivalence gate changed")
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
            "phase": "inside_union_equivalence_runtime",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure)
            }),
            "report_exists_after_attempt": report_path.exists(),
        })
        _rewrite_json_v20a(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-union-equivalence-report-v20a",
        "recipe": recipe,
        "configuration": configuration,
        "runtime_audit": audit,
        "summary": summary,
        "gate": gate,
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
        "attribution_run_opened": False,
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    validate_compact_report_v20a(
        report, expected_recipe=recipe, expected_implementation=implementation
    )
    _exclusive_write_json_v20a(report_path, report)
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
    _rewrite_json_v20a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v20a(argv)
    moe = _validate_moe_backend_environment_v20a()
    layer_bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v20a().parse_args(remaining)
    implementation = implementation_identity_v20a()
    preregistration, panel_bundle = _load_bound_inputs_v20a(layer_bundle)
    recipe = recipe_v20a(
        layer_bundle, preregistration, panel_bundle, implementation, moe
    )
    validate_runtime_v20a(args, layer_bundle, implementation, recipe)
    if args.v20a_dry_run:
        payload = {
            "schema": "eggroll-es-union-equivalence-dry-run-v20a",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "implementation": implementation,
            "recipe": recipe,
            "real_launch_requires_committed_source_bundle": True,
            "gpu_launched": False,
            "attribution_run_opened": False,
        }
        payload["content_sha256_before_self_field"] = canonical_sha256(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v20a(layer_bundle, panel_bundle, implementation, recipe)


if __name__ == "__main__":
    main()
