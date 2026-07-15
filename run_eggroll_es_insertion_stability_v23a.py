#!/usr/bin/env python3
"""Fail-closed four-model train-only runtime for V23A insertion stability."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a
import eggroll_es_worker_v23a as worker_v23a
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_insertion_stability_v23a as mechanics_v23a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V23A = prereg_v23a.OUTPUT_PATH_V23A
PREREG_FILE_SHA256_V23A = (
    "6dfdf59ed6e9be494fdbd2450eca296d2e334bfac54b46fb59309f7be62ccf57"
)
PREREG_CONTENT_SHA256_V23A = (
    "de43c14ae4fc325dfd23351fd1021dd029282b93f6cb6a17c4073c2ac72cc281"
)
PREREG_COMMIT_V23A = "fc8c42be67b832ee478eebce67b3bda375467ff6"
MECHANICS_COMMIT_V23A = "55def620b86e443c785baf0620b1f40487e49b34"
EXPERIMENT_NAME_V23A = "insertion_location_stability_v23a_authoritative_raw"
OUTPUT_DIRECTORY_V23A = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
REPORT_NAME_V23A = "insertion_location_stability_v23a.json"
ATTEMPT_NAME_V23A = f".{EXPERIMENT_NAME_V23A}.launch_attempt.json"
WORKER_EXTENSION_V23A = (
    "eggroll_es_worker_v23a.InsertionLocationAuditWorkerExtensionV23A"
)
TEST_PATH_V23A = (ROOT / "test_run_eggroll_es_insertion_stability_v23a.py").resolve()

FROZEN_PHASE_PATHS_V23A = {
    "model_seal_builder_v23a": ROOT / "build_eggroll_es_insertion_model_seal_v23a.py",
    "model_seal_tests_v23a": ROOT / "test_build_eggroll_es_insertion_model_seal_v23a.py",
    "model_seal_v23a": prereg_v23a.MODEL_SEAL_PATH_V23A,
    "prereg_module_v23a": Path(prereg_v23a.__file__).resolve(),
    "prereg_tests_v23a": ROOT / "test_eggroll_es_insertion_stability_preregistration_v23a.py",
    "preregistration_v23a": PREREG_PATH_V23A,
    "worker_v23a": Path(worker_v23a.__file__).resolve(),
    "worker_tests_v23a": ROOT / "test_eggroll_es_worker_v23a.py",
    "mechanics_v23a": Path(mechanics_v23a.__file__).resolve(),
    "mechanics_tests_v23a": ROOT / "test_train_eggroll_es_insertion_stability_v23a.py",
}
for _arm, _filename in {
    "base_middle_late": "v23a_base_middle_late_dense.json",
    "insert_front_e005": "v23a_insert_front_e005_dense.json",
    "insert_middle_e005": "v23a_insert_middle_e005_dense.json",
    "insert_back_e005": "v23a_insert_back_e005_dense.json",
}.items():
    FROZEN_PHASE_PATHS_V23A[f"layer_plan_{_arm}"] = (
        ROOT / "experiments/layer_plans" / _filename
    )

FROZEN_PHASE_HASHES_V23A = {
    "model_seal_builder_v23a": "7137b944c9afd8955d0a66bc8a1813393ab05d7815094152c0d8176af028a9de",
    "model_seal_tests_v23a": "452cfffa02e29b1fcb6f4958a1f7c4825a068851235d5d79de3ca2b7b85951aa",
    "model_seal_v23a": "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440",
    "prereg_module_v23a": "01433f77c61b1afd01fb82610e6cca93d66895b630627edeb88f875149812453",
    "prereg_tests_v23a": "ed67668a12a57e4a79e7b4415ee63645b8ebc9da277e243bcd551f66ab8b1eaf",
    "preregistration_v23a": PREREG_FILE_SHA256_V23A,
    "worker_v23a": "cf537e2df8344c40bc264f3e3f07eb2a24dfc01089e87e9432bbe0617d9c962f",
    "worker_tests_v23a": "22c28e228d7f5fa7fefe2bca3d35028fec9853b76f592336d2c034b26c67ca00",
    "mechanics_v23a": "4b3c0c4ffea6dc5c0703dbe901e025f31eccb5fb5af25a4a044b1ece3d721b83",
    "mechanics_tests_v23a": "a7aa43e7912f7009b4bcadbf9eb9acc071f65728e60aaea3ebd8cf1be28933ce",
    "layer_plan_base_middle_late": "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747",
    "layer_plan_insert_front_e005": "4c0f451545f01ec53cc17800e24f331d1038bbc3ce96fe352a9cc2b96c822e29",
    "layer_plan_insert_middle_e005": "9f343cb136a5d4883ae81878ecec005e028b7f5e492ca0cc64b1f9e1945c112a",
    "layer_plan_insert_back_e005": "21a0100d2bf729ce5ce88ea83ea668086cfb512ff5684413050d03d796c7820e",
}
IMPLEMENTATION_PATHS_V23A = {
    **driver_v13.IMPLEMENTATION_PATHS_V13,
    **FROZEN_PHASE_PATHS_V23A,
    "runtime_driver_v23a": Path(__file__).resolve(),
    "runtime_tests_v23a": TEST_PATH_V23A,
}
FORBIDDEN_ARGV_TOKENS_V23A = (
    "checkpoint", "update", "heldout", "holdout", "validation", "ood",
    "benchmark", "eval", "save-model", "train-dataset", "promotion", "union",
)
FORBIDDEN_PERSISTED_KEYS_V23A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "responses", "unit_scores",
    "bootstrap_draws", "bootstrap_replicates", "row_content", "row_sha256",
}
MOE_OVERRIDE_ENVIRONMENT_V23A = (
    "VLLM_TUNED_CONFIG_FOLDER", "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE", "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)

canonical_sha256 = prereg_v23a.canonical_sha256
file_sha256 = prereg_v23a.file_sha256
anchor_v4 = anchor_v13.anchor_v4


def _without_self_v23a(value):
    return {key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"}


def _seal_v23a(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _recursive_keys_v23a(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys_v23a(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys_v23a(item)


def _assert_compact_persistence_v23a(value):
    overlap = FORBIDDEN_PERSISTED_KEYS_V23A & set(_recursive_keys_v23a(value))
    if overlap:
        raise RuntimeError(f"v23a compact output contains forbidden keys: {sorted(overlap)}")


def _assert_train_only_argv_v23a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(item in lowered for item in FORBIDDEN_ARGV_TOKENS_V23A):
            raise ValueError(f"v23a rejects forbidden runtime surface: {token}")


def _load_preregistration_v23a():
    value = json.loads(PREREG_PATH_V23A.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V23A) != PREREG_FILE_SHA256_V23A
        or value["content_sha256_before_self_field"] != PREREG_CONTENT_SHA256_V23A
        or value["content_sha256_before_self_field"]
        != canonical_sha256(_without_self_v23a(value))
    ):
        raise RuntimeError("v23a preregistration file identity changed")
    # The immutable artifact was serialized with sort_keys=True, while its
    # validator intentionally requires the preregistered semantic arm order.
    # Rehydrate that order in memory; canonical JSON hashing is key-sorted, so
    # this does not alter the sealed content identity.
    value["arms"] = {
        arm: value["arms"][arm] for arm in prereg_v23a.ARM_ORDER_V23A
    }
    prereg_v23a.validate_preregistration_v23a(value)
    return value


def _verify_commit_file_v23a(path, commit, expected):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != expected or file_sha256(path) != expected:
        raise RuntimeError(f"v23a committed phase changed: {relative}")


def _verify_frozen_phases_v23a():
    model_keys = {"model_seal_builder_v23a", "model_seal_tests_v23a", "model_seal_v23a"}
    mechanics_keys = {"worker_v23a", "worker_tests_v23a", "mechanics_v23a", "mechanics_tests_v23a"}
    for key, path in FROZEN_PHASE_PATHS_V23A.items():
        commit = (
            prereg_v23a.MODEL_SEAL_COMMIT_V23A if key in model_keys
            else MECHANICS_COMMIT_V23A if key in mechanics_keys
            else PREREG_COMMIT_V23A
        )
        _verify_commit_file_v23a(path, commit, FROZEN_PHASE_HASHES_V23A[key])


def implementation_identity_v23a():
    inherited = driver_v13.implementation_identity_v13()
    _verify_frozen_phases_v23a()
    files = {
        key: {"path": str(Path(path).resolve()), "file_sha256": file_sha256(path)}
        for key, path in IMPLEMENTATION_PATHS_V23A.items()
    }
    if {key: files[key] for key in inherited["files"]} != inherited["files"]:
        raise RuntimeError("v23a inherited V13 implementation changed")
    expected = {key: files[key]["file_sha256"] for key in FROZEN_PHASE_HASHES_V23A}
    if expected != FROZEN_PHASE_HASHES_V23A:
        raise RuntimeError("v23a frozen foundation or mechanics changed")
    return {
        "files": files,
        "inherited_v13_bundle_sha256": inherited["bundle_sha256"],
        "frozen_phase_bundle_sha256": canonical_sha256({
            key: files[key] for key in FROZEN_PHASE_PATHS_V23A
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def _source_provenance_v23a(implementation):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"v23a real launch requires committed source: {relative}") from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v23a source differs from committed HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    return _seal_v23a({
        "schema": "eggroll-es-committed-source-bundle-v23a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    })


def recipe_v23a(preregistration, implementation):
    value = {
        "schema": "eggroll-es-insertion-location-runtime-recipe-v23a",
        "experiment_name": EXPERIMENT_NAME_V23A,
        "preregistration": {
            "path": str(PREREG_PATH_V23A),
            "file_sha256": PREREG_FILE_SHA256_V23A,
            "content_sha256": PREREG_CONTENT_SHA256_V23A,
            "commit": PREREG_COMMIT_V23A,
        },
        "arms": copy.deepcopy(preregistration["arms"]),
        "panel_contract": copy.deepcopy(preregistration["panel_contract"]),
        "fresh_basis": copy.deepcopy(preregistration["fresh_basis"]),
        "runtime": copy.deepcopy(preregistration["runtime"]),
        "analysis": copy.deepcopy(preregistration["analysis"]),
        "authority": copy.deepcopy(preregistration["authority"]),
        "worker_extension": WORKER_EXTENSION_V23A,
        "worker_update_surfaces_closed": True,
        "coordinator_update_surfaces_closed": True,
        "post_restore_probe_requests_all_engines": 4,
        "post_restore_probes_excluded_from_preregistered_endpoint_scoring": True,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(OUTPUT_DIRECTORY_V23A),
    }
    return _seal_v23a(value)


def _parser_v23a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_v23a(args, preregistration, implementation, recipe):
    prereg_v23a.validate_preregistration_v23a(preregistration)
    if any(os.environ.get(key) for key in MOE_OVERRIDE_ENVIRONMENT_V23A):
        raise ValueError("v23a rejects external MoE backend overrides")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if not args.v23a_dry_run and expected is None:
            raise ValueError(f"v23a real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v23a {label} hash changed")


def validate_live_model_directories_v23a(preregistration):
    """Rebuild the complete model seal before a real actor can be created."""
    rebuilt = prereg_v23a.model_seal_v23a.build_model_seal_v23a()
    persisted = json.loads(prereg_v23a.MODEL_SEAL_PATH_V23A.read_text(encoding="utf-8"))
    if (
        rebuilt != persisted
        or rebuilt["content_sha256_before_self_field"]
        != prereg_v23a.MODEL_SEAL_CONTENT_SHA256_V23A
        or any(
            rebuilt["arms"][arm]["path"] != preregistration["arms"][arm]["model_path"]
            or rebuilt["arms"][arm]["all_files_fingerprint_sha256"]
            != preregistration["arms"][arm]["model_directory_fingerprint_sha256"]
            for arm in prereg_v23a.ARM_ORDER_V23A
        )
    ):
        raise RuntimeError("v23a live model directories differ from the complete tensor seal")
    return {
        "model_seal_content_sha256": rebuilt["content_sha256_before_self_field"],
        "all_four_complete_tensor_seals_match": True,
    }


def _unwrap_one_v23a(value, method):
    result = anchor_v13.anchor_v4.anchor_v3._unwrap_tp1_results_v3([value], 1, method)
    if len(result) != 1:
        raise RuntimeError(f"v23a {method} returned partial TP1 result")
    return result[0]


def _score_panel_outputs_v23a(dense_items, outputs):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = np.asarray(
        [item["mean_answer_token_logprob"] for item in dense["examples"]],
        dtype=np.float64,
    )
    if rewards.shape != (56,) or not np.isfinite(rewards).all():
        raise RuntimeError("v23a train-panel dense score coverage changed")
    return rewards, canonical_sha256(dense)


def _validate_device_identities_v23a(reports):
    if len(reports) != 4:
        raise RuntimeError("v23a device identity coverage is incomplete")
    visible = []
    for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
        report = reports[rank]
        token = report.get("cuda_visible_devices") if isinstance(report, dict) else None
        if (
            report.get("schema") != "eggroll-es-runtime-device-identity-v23a"
            or report.get("rank") != rank or report.get("world_size") != 4
            or report.get("arm") != arm or report.get("runtime_cuda_device") != 0
            or report.get("update_surfaces_closed") is not True
            or token != str(rank)
            or not isinstance(report.get("device_total_memory"), int)
            or report["device_total_memory"] <= 0
        ):
            raise RuntimeError("v23a fixed rank-to-physical-GPU assignment changed")
        visible.append(token)
    if len(set(visible)) != 4:
        raise RuntimeError("v23a engines do not occupy four distinct GPUs")
    return {"reports_sha256": canonical_sha256(reports), "gpu_ids": [0, 1, 2, 3]}


def _expected_plan_binding_v23a(arm, preregistration):
    plan = preregistration["arms"][arm]["layer_plan"]
    frozen = worker_v23a.FROZEN_LAYER_PLANS_V23A[plan["plan_sha256"]]
    return {
        "layer_plan_file_sha256": plan["file_sha256"],
        "layer_plan_sha256": plan["plan_sha256"],
        "checkpoint_to_runtime_mapping_sha256": frozen["checkpoint_to_runtime_mapping_sha256"],
        "source_unit_count": frozen["source_unit_count"],
        "runtime_selected_name_sha256": frozen["runtime_selected_name_sha256"],
        "runtime_selected_parameter_count": frozen["runtime_selected_parameter_count"],
        "selected_element_count": frozen["selected_element_count"],
        "dense_reward_sha256": anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
    }


def _validate_cross_arm_logical_contract_v23a(preregistration):
    plans = [
        preregistration["arms"][arm]["layer_plan"]
        for arm in prereg_v23a.ARM_ORDER_V23A
    ]
    contract = preregistration["cross_arm_perturbation_contract"]
    logical = {item["logical_shape_order_sha256"] for item in plans}
    if (
        len(logical) != 1
        or logical.pop() != contract["logical_shape_order_sha256"]
        or any(item["num_units"] != 35 for item in plans)
        or any(item["selected_element_count"] != 142_999_552 for item in plans)
        or contract != {
            "logical_shape_order_sha256": contract["logical_shape_order_sha256"],
            "source_unit_count_per_arm": 35,
            "runtime_selected_parameter_count_per_arm": 23,
            "selected_element_count_per_arm": 142_999_552,
            "same_logical_tensor_shape_and_order_all_arms": True,
            "same_direction_seed_all_arms_every_signed_wave": True,
        }
    ):
        raise RuntimeError("v23a cross-arm logical perturbation geometry changed")
    return contract["logical_shape_order_sha256"]


def _validate_perturbation_results_v23a(results):
    if (
        not isinstance(results, list) or len(results) != 4
        or any(not isinstance(item, (list, tuple)) or list(item) != [None]
               for item in results)
    ):
        raise RuntimeError("v23a perturbation wave did not complete on all four TP1 workers")


def _validate_install_v23a(report, arm, rank, preregistration):
    binding = _expected_plan_binding_v23a(arm, preregistration)
    frozen = worker_v23a.FROZEN_LAYER_PLANS_V23A[binding["layer_plan_sha256"]]
    if (
        not isinstance(report, dict)
        or report.get("schema") != "eggroll-es-layer-plan-installed-v4"
        or report.get("installed") is not True or report.get("idempotent") is not False
        or report.get("rank") != rank or report.get("world_size") != 4
        or report.get("plan") != frozen["plan"]
        or report.get("selected_byte_count") != frozen["selected_byte_count"]
        or any(report.get(key) != value for key, value in binding.items())
        or not isinstance(report.get("selected_parameter_manifest_sha256"), str)
        or not isinstance(report.get("unselected_origin_sha256"), str)
    ):
        raise RuntimeError(f"v23a layer-plan installation changed for {arm}")
    return {**binding,
            "selected_parameter_manifest_sha256": report["selected_parameter_manifest_sha256"],
            "unselected_origin_sha256": report["unselected_origin_sha256"]}


class InsertionLocationRuntimeMixinV23A:
    """Score four sealed model arms without exposing an update entrypoint."""

    def configure_insertion_stability_v23a(self, preregistration, panel_bundle):
        prereg_v23a.validate_preregistration_v23a(preregistration)
        anchor_v13.validate_panel_bundle_v13(panel_bundle)
        if (
            len(self.engines) != 4 or self.n_vllm_engines != 4
            or self.n_gpu_per_vllm_engine != 1 or self.population_size != 32
            or not math.isclose(float(self.sigma), 0.0003, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RuntimeError("v23a requires the exact four-arm TP1 alpha-zero recipe")
        logical_shape_order = _validate_cross_arm_logical_contract_v23a(preregistration)
        device_handles = [
            engine.collective_rpc.remote("runtime_device_identity_v23a", args=(arm,))
            for engine, arm in zip(self.engines, prereg_v23a.ARM_ORDER_V23A)
        ]
        device_reports = [
            _unwrap_one_v23a(item, "runtime_device_identity_v23a")
            for item in self._resolve(device_handles)
        ]
        device_identity = _validate_device_identities_v23a(device_reports)

        install_handles = []
        for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
            plan = preregistration["arms"][arm]["layer_plan"]
            install_handles.append(self.engines[rank].collective_rpc.remote(
                "install_layer_plan_v4",
                args=(
                    Path(plan["path"]).read_bytes(), plan["file_sha256"],
                    plan["plan_sha256"], anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                    anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
                ),
            ))
        install_reports = [
            _unwrap_one_v23a(item, "install_layer_plan_v4")
            for item in self._resolve(install_handles)
        ]
        installs = {
            arm: _validate_install_v23a(install_reports[rank], arm, rank, preregistration)
            for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A)
        }
        references_raw = self._resolve([
            engine.collective_rpc.remote("save_self_exact_reference", args=())
            for engine in self.engines
        ])
        references = [
            _unwrap_one_v23a(item, "save_self_exact_reference")
            for item in references_raw
        ]
        states_raw = self._resolve([
            engine.collective_rpc.remote(
                "inspect_cached_distributed_update_state_v4", args=(4, "exact_reference")
            ) for engine in self.engines
        ])
        states = [
            _unwrap_one_v23a(item, "inspect_cached_distributed_update_state_v4")
            for item in states_raw
        ]
        compact_references = {}
        for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
            reference, state, install = references[rank], states[rank], installs[arm]
            communicator = state.get("communicator", {})
            if (
                reference.get("reference_generation") != 1
                or reference.get("fresh_for_population") is not True
                or state.get("reference_generation") != 1
                or state.get("reference_fresh_for_population") is not True
                or state.get("pending") is not False
                or state.get("reference_identity") != reference.get("identity")
                or state.get("current_identity") != reference.get("identity")
                or communicator.get("rank") != rank or communicator.get("world_size") != 4
                or communicator.get("tp_world_size") != 1
                or any(state.get(key) != value for key, value in install.items())
            ):
                raise RuntimeError(f"v23a exact reference capture changed for {arm}")
            compact_references[arm] = {
                "reference_generation": 1,
                "reference_sha256": reference["identity"]["sha256"],
                "reference_identity": reference["identity"],
            }
        self._v23a_preregistration = copy.deepcopy(preregistration)
        self._v23a_panel_bundle = copy.deepcopy(panel_bundle)
        self._v23a_installs = installs
        self._v23a_references = compact_references
        return {
            "schema": "eggroll-es-insertion-location-runtime-configuration-v23a",
            "device_identity": device_identity,
            "installations_sha256": canonical_sha256(installs),
            "references_sha256": canonical_sha256(compact_references),
            "cross_arm_logical_shape_order_sha256": logical_shape_order,
            "panel_bundle_content_sha256": panel_bundle["content_sha256_before_self_field"],
            "four_distinct_tp1_model_arms": True,
            "update_surfaces_closed": True,
        }

    def _prepared_train_requests_v23a(self):
        anchor_v13.validate_panel_bundle_v13(self._v23a_panel_bundle)
        panels = {}
        requests = []
        cursor = 0
        for panel_name in mechanics_v23a.PANEL_NAMES_V23A:
            panel = self._v23a_panel_bundle["panels"][panel_name]
            prompts = [base.specialist_template(item) for item in panel["questions"]]
            dense_items = anchor_v4.prepare_gold_answer_items_v4(
                self.tokenizer, prompts, panel["answers"]
            )
            if len(dense_items) != 56:
                raise RuntimeError("v23a fixed panel request count changed")
            panels[panel_name] = {"slice": (cursor, cursor + 56), "dense_items": dense_items}
            requests.extend({"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items)
            cursor += 56
        if cursor != 280:
            raise RuntimeError("v23a fixed signed-wave request count changed")
        return panels, requests, canonical_sha256(requests)

    def _generate_all_v23a(self, requests):
        batches = self._resolve([
            engine.generate.remote(
                list(requests), self._dense_sampling_params_v4(0), use_tqdm=False
            ) for engine in self.engines
        ])
        if not isinstance(batches, list) or len(batches) != 4 or any(
            not isinstance(batch, list) or len(batch) != len(requests) for batch in batches
        ):
            raise RuntimeError("v23a all-four-engine generation was incomplete")
        return batches

    def _score_batches_v23a(self, panels, batches):
        scores = {}
        commitments = {}
        probes = {}
        for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
            arm_scores = np.empty((5, 56), dtype=np.float64)
            arm_hashes = []
            for panel_index, panel_name in enumerate(mechanics_v23a.PANEL_NAMES_V23A):
                panel = panels[panel_name]
                start, stop = panel["slice"]
                rewards, digest = _score_panel_outputs_v23a(
                    panel["dense_items"], batches[rank][start:stop]
                )
                arm_scores[panel_index] = rewards
                arm_hashes.append(digest)
            first_panel = panels[mechanics_v23a.PANEL_NAMES_V23A[0]]
            probe_dense = anchor_v4.score_gold_answer_outputs_v4(
                [first_panel["dense_items"][0]], [batches[rank][0]]
            )
            scores[arm] = arm_scores
            commitments[arm] = arm_hashes
            probes[arm] = canonical_sha256(probe_dense)
        return scores, commitments, probes

    def _restore_verify_v23a(self):
        restored_raw = self._resolve([
            engine.collective_rpc.remote("restore_self_weights_exact", args=())
            for engine in self.engines
        ])
        restored = [_unwrap_one_v23a(item, "restore_self_weights_exact") for item in restored_raw]
        if restored != [True] * 4:
            raise RuntimeError("v23a exact selected restore was incomplete")
        checks_raw = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        checks = [_unwrap_one_v23a(item, "verify_self_exact_reference") for item in checks_raw]
        if any(item.get("passed") is not True for item in checks):
            raise RuntimeError("v23a selected-reference verification failed")
        return canonical_sha256(checks)

    def _boundary_audit_v23a(self):
        handles = []
        for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
            reference = self._v23a_references[arm]
            handles.append(self.engines[rank].collective_rpc.remote(
                "audit_population_completion_v4",
                args=(4, reference["reference_generation"], reference["reference_sha256"]),
            ))
        reports = [
            _unwrap_one_v23a(item, "audit_population_completion_v4")
            for item in self._resolve(handles)
        ]
        for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
            report, reference, install = reports[rank], self._v23a_references[arm], self._v23a_installs[arm]
            if (
                report.get("schema") != "eggroll-es-post-population-audit-v4"
                or report.get("passed") is not True or report.get("rank") != rank
                or report.get("world_size") != 4
                or report.get("reference_generation") != 1
                or report.get("reference_sha256") != reference["reference_sha256"]
                or report.get("current_identity") != reference["reference_identity"]
                or any(report.get(key) != value for key, value in install.items())
            ):
                raise RuntimeError(f"v23a full post-population audit failed for {arm}")
        return {"passed": True, "reports_sha256": canonical_sha256(reports)}

    def estimate_insertion_stability_v23a(self):
        schedule = prereg_v23a.signed_wave_schedule_v23a()
        panels, requests, request_identity = self._prepared_train_requests_v23a()
        reference_batches = self._generate_all_v23a(requests)
        reference_scores, reference_commitments, pre_probes = self._score_batches_v23a(
            panels, reference_batches
        )
        unit_scores = {
            arm: np.full((5, 2, 32, 56), np.nan, dtype=np.float64)
            for arm in prereg_v23a.ARM_ORDER_V23A
        }
        dense_commitments = []
        restore_hashes = []
        for item in schedule:
            perturbed = self._resolve([
                engine.collective_rpc.remote(
                    "perturb_self_weights",
                    args=(item["direction_seed"], 0.0003, item["negate"]),
                ) for engine in self.engines
            ])
            _validate_perturbation_results_v23a(perturbed)
            batches = self._generate_all_v23a(requests)
            wave_scores, wave_commitments, _unused_probes = self._score_batches_v23a(
                panels, batches
            )
            sign_index = 0 if item["sign"] == "plus" else 1
            for arm in prereg_v23a.ARM_ORDER_V23A:
                unit_scores[arm][:, sign_index, item["direction_index"], :] = wave_scores[arm]
                dense_commitments.extend(wave_commitments[arm])
            restore_hashes.append(self._restore_verify_v23a())
        if (
            any(not np.isfinite(values).all() for values in unit_scores.values())
            or len(dense_commitments) != 1280 or len(restore_hashes) != 64
        ):
            raise RuntimeError("v23a signed population capture is incomplete")
        boundary = self._boundary_audit_v23a()
        first_request = [requests[0]]
        post_batches = self._generate_all_v23a(first_request)
        first_panel = panels[mechanics_v23a.PANEL_NAMES_V23A[0]]
        post_probes = {
            arm: canonical_sha256(anchor_v4.score_gold_answer_outputs_v4(
                [first_panel["dense_items"][0]], [post_batches[rank][0]]
            )) for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A)
        }
        if pre_probes != post_probes:
            raise RuntimeError("v23a pre/post unperturbed reference probes drifted")
        compact = mechanics_v23a.build_compact_estimator_summary_v23a(
            unit_scores, reference_scores, self._v23a_panel_bundle
        )
        unit_scores = None
        reference_scores = None
        integrity_template = {
            "all_sixty_four_signed_waves_complete": True,
            "all_five_panels_every_arm_and_signed_wave": True,
            "same_direction_seed_all_four_arms": True,
            "same_fixed_requests_all_four_arms": True,
            "exact_selected_reference_restored_every_signed_wave": True,
            "unselected_origin_unchanged": boundary["passed"],
            "pre_post_unperturbed_reference_probe_equal": True,
            "distinct_gpu_assignment_verified": True,
            "union_planner_called": False,
            "all_integrity_audits_passed": boundary["passed"],
        }
        compact.pop("content_sha256_before_self_field", None)
        compact["runtime_integrity"] = {
            arm: dict(integrity_template) for arm in prereg_v23a.ARM_ORDER_V23A
        }
        _seal_v23a(compact)
        gate = mechanics_v23a.evaluate_gate_v23a(compact)
        audit = _seal_v23a({
            "schema": "eggroll-es-insertion-location-runtime-compact-audit-v23a",
            "fixed_request_identity_sha256": request_identity,
            "reference_dense_commitments_sha256": canonical_sha256(reference_commitments),
            "pre_post_probe_identity_sha256": canonical_sha256({"pre": pre_probes, "post": post_probes}),
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "dense_result_commitments_sha256": canonical_sha256(dense_commitments),
            "population_boundary_audit_sha256": boundary["reports_sha256"],
            "signed_wave_count": 64,
            "requests_per_engine_per_signed_wave": 280,
            "requests_all_engines_per_signed_wave": 1120,
            "requests_per_engine_all_signed_waves": 17920,
            "requests_all_engines_all_signed_waves": 71680,
            "unperturbed_reference_requests_all_engines": 1120,
            "post_restore_probe_requests_all_engines": 4,
            "dense_result_commitment_count": 1280,
            "per_unit_scores_persisted": False,
            "bootstrap_replicates_persisted": False,
            "model_update_applied": False,
        })
        _assert_compact_persistence_v23a({"estimator": compact, "gate": gate, "audit": audit})
        return compact, gate, audit

    @staticmethod
    def _closed_surface_v23a(*_args, **_kwargs):
        raise RuntimeError("v23a closes all update checkpoint evaluation and union surfaces")

    configure_anchor = _closed_surface_v23a
    configure_train_panels_v13 = _closed_surface_v23a
    estimate_train_panels_v13 = _closed_surface_v23a
    estimate_step_coefficients = _closed_surface_v23a
    apply_seed_coefficients = _closed_surface_v23a
    train_step = _closed_surface_v23a
    evaluate_handle = _closed_surface_v23a
    evaluate_population_on_batch = _closed_surface_v23a
    eval_step = _closed_surface_v23a
    fit = _closed_surface_v23a


def load_runtime_trainer_v23a(preregistration):
    prereg_v23a.validate_preregistration_v23a(preregistration)
    base_plan = preregistration["arms"]["base_middle_late"]["layer_plan"]
    bundle = anchor_v4.load_frozen_layer_plan_v4(
        base_plan["path"], expected_file_sha256=base_plan["file_sha256"],
        expected_plan_sha256=base_plan["plan_sha256"],
        expected_model_config_sha256=base_plan["model_config_sha256"],
    )
    # V13 contributes the frozen train panels, but its trainer validator only
    # admits the historical 8-layer production plans.  Start from V4 so this
    # runtime can install the four preregistered 4-layer V23A plans itself.
    parent = anchor_v4.load_trainer(bundle)

    class InsertionLocationRuntimeTrainerV23A(InsertionLocationRuntimeMixinV23A, parent):
        def launch_engines(self, num_engines=4, n_gpu_per_vllm_engine=1,
                           model_name="unused", precision="bfloat16"):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("v23a requires exactly four TP1 engines")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            pgs = [placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK", lifetime="detached")
                   for _ in range(4)]
            ray.get([pg.ready() for pg in pgs])
            strategies = [PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ) for pg in pgs]
            engines = []
            for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
                args = {
                    "model": preregistration["arms"][arm]["model_path"],
                    "tensor_parallel_size": 1,
                    "worker_extension_cls": WORKER_EXTENSION_V23A,
                    "dtype": precision, "enable_prefix_caching": False,
                    "enforce_eager": True, "gpu_memory_utilization": 0.82,
                    "max_model_len": 2048,
                    "limit_mm_per_prompt": {"image": 0, "video": 0},
                    "mm_processor_cache_gb": 0, "skip_mm_profiling": True,
                    "moe_backend": "triton",
                }
                engines.append(ray.remote(
                    num_cpus=0, num_gpus=1, scheduling_strategy=strategies[rank]
                )(ESNcclLLM).remote(**args))
            return engines, pgs

    return InsertionLocationRuntimeTrainerV23A


def _make_trainer_v23a(preregistration):
    cls = load_runtime_trainer_v23a(preregistration)
    model = preregistration["arms"]["base_middle_late"]["model_path"]
    return cls(
        model_name=model, checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=56, mini_batch_size=56,
        reward_function=base.specialist_reward, template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_V23A,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(OUTPUT_DIRECTORY_V23A),
    )


def _exclusive_write_json_v23a(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_seal_v23a(value), indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"v23a exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(raw); output.flush(); os.fsync(output.fileno())


def _rewrite_json_v23a(path, value):
    _seal_v23a(value)
    driver_v13.driver_v1.atomic_write_json(path, value)


def run_exact_v23a(preregistration, implementation, recipe):
    attempt_path = OUTPUT_DIRECTORY_V23A / ATTEMPT_NAME_V23A
    run_dir = OUTPUT_DIRECTORY_V23A / EXPERIMENT_NAME_V23A
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v23a requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v23a(implementation)
    model_directory_audit = validate_live_model_directories_v23a(preregistration)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v23a", "status": "launching",
        "phase": "before_trainer_creation", "recipe": recipe,
        "source_provenance": provenance, "model_update_applied": False,
        "model_directory_audit": model_directory_audit,
        "nontrain_surface_opened": False,
    }
    _exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({"status": "failed", "phase": "fresh_run_reservation_race"})
        _rewrite_json_v23a(attempt_path, attempt)
        raise RuntimeError("v23a run directory appeared after exclusive attempt claim")
    trainer = None
    failure = None
    failure_traceback = None
    result = None
    configured = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v23a(preregistration)
        panels = mechanics_v23a.load_panel_bundle_v23a()
        configured = trainer.configure_insertion_stability_v23a(preregistration, panels)
        result = trainer.estimate_insertion_stability_v23a()
    except BaseException as error:
        failure = error; failure_traceback = traceback.format_exc()
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error; failure_traceback = traceback.format_exc()
                else:
                    failure_traceback += "\nCleanup failure:\n" + traceback.format_exc()
    if failure is not None:
        attempt.update({
            "status": "failed", "phase": "inside_v23a_train_only_runtime",
            "failure": {"type": type(failure).__name__, "message": str(failure),
                        "traceback": failure_traceback},
            "model_update_applied": False,
        })
        _rewrite_json_v23a(attempt_path, attempt)
        raise failure
    estimator, gate, audit = result
    report = {
        "schema": "eggroll-es-insertion-location-stability-report-v23a",
        "recipe": recipe, "configuration": configured, "estimator": estimator,
        "gate": gate, "runtime_audit": audit, "implementation": implementation,
        "model_update_applied": False, "nontrain_surface_opened": False,
        "direct_action_taken": False,
    }
    _assert_compact_persistence_v23a(report)
    report_path = run_dir / REPORT_NAME_V23A
    _exclusive_write_json_v23a(report_path, report)
    attempt.update({
        "status": "complete", "phase": "after_cleanup_and_compact_report",
        "report_binding": {"path": str(report_path), "file_sha256": file_sha256(report_path),
                           "content_sha256": report["content_sha256_before_self_field"]},
        "model_update_applied": False,
    })
    _rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v23a(argv)
    args = _parser_v23a().parse_args(argv)
    preregistration = _load_preregistration_v23a()
    implementation = implementation_identity_v23a()
    recipe = recipe_v23a(preregistration, implementation)
    validate_runtime_v23a(args, preregistration, implementation, recipe)
    if args.v23a_dry_run:
        payload = _seal_v23a({
            "schema": "eggroll-es-insertion-location-launch-dry-run-v23a",
            "recipe": recipe, "implementation": implementation,
            "real_launch_requires_committed_bundle_and_recipe_hashes": True,
            "gpu_launched": False,
        })
        _assert_compact_persistence_v23a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v23a(preregistration, implementation, recipe)


if __name__ == "__main__":
    main()
