#!/usr/bin/env python3
"""Fail-closed four-GPU train-only lagged-replay calibration for V35A.

This runtime scores only the three frozen V13 optimization panels.  It has no
model-update, checkpoint, evaluation, dataset-promotion, or replay-HPO surface.
The only scientific output is a content-free provisional manual-review pool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import eggroll_es_train_panel_sampler_v13 as sampler_v13
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_eggroll_es_paired_data_compat_v33a as runtime_v33a


ROOT = Path(__file__).resolve().parent
PREREG_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_LAGGED_REPLAY_CALIBRATION_V35A_PREREGISTRATION.json"
).resolve()
PROTOCOL_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_LAGGED_REPLAY_CALIBRATION_V35A_PROTOCOL.md"
).resolve()
MANIFEST_PATH = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
).resolve()
TEST_PATH = (
    ROOT / "test_run_eggroll_es_lagged_replay_calibration_v35a.py"
).resolve()

PREREG_FILE_SHA256 = (
    "fcfaead965d16173324c779112243ba8eb5472a806b09647a5eb0049b1589561"
)
PREREG_CONTENT_SHA256 = (
    "3a5240d6863a4d65f43ffdcbfe2c7b68e11fbb1f8cfc51b0e4c16b261601b094"
)
PROTOCOL_FILE_SHA256 = (
    "e1cda725207239f55c2b59df15777de6f95519e823ab6110e340ab4d2a008bb2"
)
MANIFEST_FILE_SHA256 = (
    "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7"
)
MANIFEST_CONTENT_SHA256 = (
    "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
)
SOURCE_FILE_SHA256 = sampler_v13.SOURCE_SHA256
SOURCE_ROWS = sampler_v13.SOURCE_ROWS
OPTIMIZATION_PANELS = ("optimization_0", "optimization_1", "optimization_2")
SCREEN_PANELS = ("train_screen_0", "train_screen_1")
STRATA = tuple(sampler_v13.STRATA)
PANEL_ROWS = 56
UNION_ROWS = 168
ENGINE_COUNT = 4
PHASES = ("A", "B", "C")
REQUESTS_PER_PHASE = ENGINE_COUNT * UNION_ROWS
PHYSICAL_REQUESTS = len(PHASES) * REQUESTS_PER_PHASE
WORKER_EXTENSION = runtime_v33a.WORKER_EXTENSION_V33A

EXPERIMENT_NAME = "s6_v35a_lagged_replay_calibration_train_only"
OUTPUT_DIRECTORY = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_PATH = OUTPUT_DIRECTORY / f".{EXPERIMENT_NAME}.launch_attempt.json"
RUN_DIRECTORY = OUTPUT_DIRECTORY / EXPERIMENT_NAME
REPORT_PATH = RUN_DIRECTORY / "lagged_replay_calibration_report_v35a.json"

ALLOWED_UNTRACKED_PREFIXES = ("experiments/dataset_probes/",)
ALLOWED_UNTRACKED_PATHS = frozenset({
    "experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl",
})
CURATOR_UNTRACKED_PATTERN = re.compile(
    r"data/manual_reviews/context_merit_audit_v([0-9]+)/.+\Z"
)
FORBIDDEN_ARGV_TOKENS = (
    "checkpoint", "update", "heldout", "holdout", "validation", "ood",
    "benchmark", "eval", "save-model", "promotion", "replay-fraction",
)
FORBIDDEN_PERSISTED_KEYS = {
    "question", "questions", "answer", "answers", "document", "documents",
    "text", "texts", "prompt", "prompts", "prompt_token_ids", "tokens",
    "token_ids", "logprobs", "outputs", "responses", "raw_scores", "scores",
    "score", "row_content", "row_index", "pid", "pids", "process_id",
    "memory_used", "gpu_memory_used", "manual_review_text",
}

canonical_sha256 = sampler_v13.canonical_sha256
file_sha256 = sampler_v13.file_sha256
anchor_v4 = runtime_v23a.anchor_v4
_seal = runtime_v23a._seal_v23a


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def assert_content_free(value):
    overlap = FORBIDDEN_PERSISTED_KEYS & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(
            f"V35A compact artifact contains forbidden keys: {sorted(overlap)}"
        )
    def validate_row_sha_records(item):
        if isinstance(item, dict):
            if "row_sha256" in item and set(item) != {
                "row_sha256", "panel", "stratum", "calibration_rank",
            }:
                raise RuntimeError(
                    "V35A row identity may persist only in the provisional-pool schema"
                )
            for child in item.values():
                validate_row_sha_records(child)
        elif isinstance(item, list):
            for child in item:
                validate_row_sha_records(child)
    validate_row_sha_records(value)


def assert_train_only_argv(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(forbidden in lowered for forbidden in FORBIDDEN_ARGV_TOKENS):
            raise ValueError(f"V35A rejects forbidden runtime surface: {token}")


def load_preregistration():
    if (
        file_sha256(PREREG_PATH) != PREREG_FILE_SHA256
        or file_sha256(PROTOCOL_PATH) != PROTOCOL_FILE_SHA256
    ):
        raise RuntimeError("V35A frozen preregistration or protocol changed")
    value = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if (
        value.get("schema")
        != "eggroll-es-lagged-replay-calibration-preregistration-v35a"
        or value.get("status")
        != "preregistered_calibration_only_no_replay_HPO_or_update_authority"
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256
        or canonical_sha256(_without_self(value)) != PREREG_CONTENT_SHA256
    ):
        raise RuntimeError("V35A preregistration content changed")
    if value["bound_inputs"]["files"]["protocol_v35a"]["file_sha256"] != (
        PROTOCOL_FILE_SHA256
    ):
        raise RuntimeError("V35A protocol binding changed")
    for item in value["bound_inputs"]["files"].values():
        if file_sha256(Path(item["path"]).resolve()) != item["file_sha256"]:
            raise RuntimeError("V35A immutable bound input changed")
    return value


def load_manifest(preregistration):
    if file_sha256(MANIFEST_PATH) != MANIFEST_FILE_SHA256:
        raise RuntimeError("V35A V13 manifest file changed")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sampler_v13.validate_manifest(manifest)
    if (
        manifest.get("content_sha256_before_self_field")
        != MANIFEST_CONTENT_SHA256
        or preregistration["bound_inputs"]["panel_manifest_content_sha256"]
        != MANIFEST_CONTENT_SHA256
        or manifest.get("source", {}).get("jsonl_sha256")
        != SOURCE_FILE_SHA256
        or manifest.get("source", {}).get("rows") != SOURCE_ROWS
    ):
        raise RuntimeError("V35A V13 manifest content binding changed")
    by_name = {panel["name"]: panel for panel in manifest["panels"]}
    if set(by_name) != set(OPTIMIZATION_PANELS + SCREEN_PANELS):
        raise RuntimeError("V35A V13 panel coverage changed")
    contract = preregistration["panels"]["contract"]
    for name, panel in by_name.items():
        expected = contract[name]
        if (
            panel["role"] != expected["role"]
            or panel["rows"] != expected["rows"]
            or panel["stratum_counts"] != expected["stratum_counts"]
            or panel["ordered_row_identity_sha256"]
            != expected["ordered_row_identity_sha256"]
        ):
            raise RuntimeError("V35A preregistered panel contract changed")
    return manifest


def optimization_metadata(manifest):
    """Project only the frozen optimization metadata; never return screens."""
    by_name = {panel["name"]: panel for panel in manifest["panels"]}
    projected = []
    seen_indices = set()
    for panel_name in OPTIMIZATION_PANELS:
        panel = by_name[panel_name]
        if panel["role"] != "optimization" or len(panel["items"]) != PANEL_ROWS:
            raise RuntimeError("V35A optimization panel geometry changed")
        for position, item in enumerate(panel["items"]):
            if item["position"] != position or item["row_index"] in seen_indices:
                raise RuntimeError("V35A optimization projection is not disjoint")
            seen_indices.add(item["row_index"])
            projected.append({
                "panel": panel_name,
                "position": position,
                "row_index": item["row_index"],
                "row_sha256": item["row_sha256"],
                "fact_id": item["fact_id"],
                "document_sha256": item["document_sha256"],
                "stratum": item["stratum"],
            })
    if len(projected) != UNION_ROWS or len(seen_indices) != UNION_ROWS:
        raise RuntimeError("V35A optimization union changed")
    screen_indices = {
        item["row_index"]
        for panel in manifest["panels"] if panel["name"] in SCREEN_PANELS
        for item in panel["items"]
    }
    if seen_indices & screen_indices:
        raise RuntimeError("V35A optimization indices overlap a frozen screen")
    return projected


def stream_exact_optimization_rows(
    source_path, metadata, *, expected_sha256, expected_rows,
):
    """Decode only selected optimization lines from the hash-bound JSONL."""
    source_path = Path(source_path).resolve()
    lowered = str(source_path).lower()
    if any(token in lowered for token in ("heldout", "holdout", "validation", "ood")):
        raise ValueError("V35A accepts only its frozen training source")
    if file_sha256(source_path) != expected_sha256:
        raise RuntimeError("V35A frozen training source changed")
    by_index = {item["row_index"]: item for item in metadata}
    if len(by_index) != UNION_ROWS:
        raise RuntimeError("V35A selected source index coverage changed")
    selected = {}
    line_count = 0
    with source_path.open("rb") as source:
        for row_index, raw_line in enumerate(source):
            line_count += 1
            if row_index not in by_index:
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("V35A selected training row is invalid JSON") from error
            item = by_index[row_index]
            required = ("question", "answer", "fact_id", "document_sha256")
            if (
                not isinstance(row, dict)
                or any(not isinstance(row.get(key), str) or not row[key] for key in required)
                or row["fact_id"] != item["fact_id"]
                or row["document_sha256"] != item["document_sha256"]
                or sampler_v13.row_sha256(row) != item["row_sha256"]
            ):
                raise RuntimeError("V35A selected training row identity changed")
            selected[row_index] = row
    if line_count != expected_rows or set(selected) != set(by_index):
        raise RuntimeError("V35A frozen source row coverage changed")
    ordered = []
    for item in metadata:
        row = selected[item["row_index"]]
        ordered.append({
            **item,
            "question": row["question"],
            "answer": row["answer"],
        })
    return ordered


def materialize_optimization_union(preregistration, manifest):
    metadata = optimization_metadata(manifest)
    source = manifest["source"]
    rows = stream_exact_optimization_rows(
        source["path"], metadata,
        expected_sha256=SOURCE_FILE_SHA256,
        expected_rows=SOURCE_ROWS,
    )
    panel_identities = {}
    for panel_name in OPTIMIZATION_PANELS:
        identities = [
            item["row_sha256"] for item in rows if item["panel"] == panel_name
        ]
        panel_identities[panel_name] = canonical_sha256(identities)
        if panel_identities[panel_name] != (
            preregistration["panels"]["contract"][panel_name][
                "ordered_row_identity_sha256"
            ]
        ):
            raise RuntimeError("V35A ordered optimization identity changed")
    audit = _seal({
        "schema": "eggroll-es-v35a-source-projection-audit",
        "source_file_sha256": SOURCE_FILE_SHA256,
        "source_row_count": SOURCE_ROWS,
        "selected_optimization_row_count": UNION_ROWS,
        "selected_index_set_sha256": canonical_sha256(
            sorted(item["row_index"] for item in metadata)
        ),
        "ordered_panel_identity_sha256": canonical_sha256(panel_identities),
        "screen_source_lines_decoded": False,
        "screen_rows_materialized_as_requests": False,
    })
    assert_content_free(audit)
    return rows, audit


def load_layer_bundle(preregistration):
    plan = preregistration["frozen_recipe"]["layer_plan"]
    return anchor_v4.load_frozen_layer_plan_v4(
        plan["path"],
        expected_file_sha256=plan["file_sha256"],
        expected_plan_sha256=plan["content_sha256"],
        expected_model_config_sha256=preregistration["frozen_recipe"][
            "model_config_sha256"
        ],
    )


def validate_live_model(preregistration):
    model = Path(preregistration["frozen_recipe"]["model"]).resolve()
    if (
        file_sha256(model / "config.json")
        != preregistration["frozen_recipe"]["model_config_sha256"]
        or file_sha256(model / "model.safetensors.index.json")
        != preregistration["frozen_recipe"]["model_index_sha256"]
    ):
        raise RuntimeError("V35A live Qwen3.6-35B-A3B identity changed")
    return _seal({
        "schema": "eggroll-es-v35a-live-model-audit",
        "config_sha256": preregistration["frozen_recipe"]["model_config_sha256"],
        "index_sha256": preregistration["frozen_recipe"]["model_index_sha256"],
    })


def implementation_identity():
    inherited = runtime_v33a.implementation_identity_v33a()
    overlay_paths = {
        "runtime_v35a": Path(__file__).resolve(),
        "runtime_tests_v35a": TEST_PATH,
        "preregistration_v35a": PREREG_PATH,
        "protocol_v35a": PROTOCOL_PATH,
        "panel_manifest_v13": MANIFEST_PATH,
        "sampler_v13": Path(sampler_v13.__file__).resolve(),
    }
    files = dict(inherited["files"])
    for key, path in overlay_paths.items():
        files[key] = {"path": str(path), "file_sha256": file_sha256(path)}
    return {
        "files": files,
        "inherited_v33a_bundle_sha256": inherited["bundle_sha256"],
        "v35a_overlay_sha256": canonical_sha256({
            key: files[key] for key in overlay_paths
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_v35a(preregistration, manifest, layer_bundle, implementation):
    value = {
        "schema": "eggroll-es-lagged-replay-calibration-runtime-recipe-v35a",
        "experiment_name": EXPERIMENT_NAME,
        "preregistration_file_sha256": PREREG_FILE_SHA256,
        "preregistration_content_sha256": PREREG_CONTENT_SHA256,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "source_row_count": SOURCE_ROWS,
        "model_config_sha256": preregistration["frozen_recipe"][
            "model_config_sha256"
        ],
        "model_index_sha256": preregistration["frozen_recipe"][
            "model_index_sha256"
        ],
        "layer_plan_file_sha256": layer_bundle["file_sha256"],
        "layer_plan_content_sha256": layer_bundle["plan_sha256"],
        "optimization_panels": [
            {
                "name": name,
                "row_count": PANEL_ROWS,
                "ordered_row_identity_sha256": next(
                    panel["ordered_row_identity_sha256"]
                    for panel in manifest["panels"] if panel["name"] == name
                ),
            }
            for name in OPTIMIZATION_PANELS
        ],
        "screen_firewall": {
            "excluded_panels": list(SCREEN_PANELS),
            "decoded_into_requests": False,
            "generated": False,
            "scored": False,
            "ranked": False,
            "selected": False,
        },
        "scoring": {
            "reward": "mean_gold_answer_token_logprob_teacher_forced_dense",
            "reward_config_sha256": anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
            "maximum_total_prompt_and_gold_answer_tokens": 1024,
        },
        "integrity_phases": {
            "ordered": list(PHASES),
            "calibration_estimand_phase": "B",
            "A_and_C_integrity_only": True,
            "exact_cross_engine_and_cross_phase_dense_commitments": True,
        },
        "request_accounting": {
            "requests_per_engine_per_phase": UNION_ROWS,
            "engines": ENGINE_COUNT,
            "requests_per_phase": REQUESTS_PER_PHASE,
            "calibration_estimand_requests": REQUESTS_PER_PHASE,
            "integrity_repeat_requests": 2 * REQUESTS_PER_PHASE,
            "physical_total_requests": PHYSICAL_REQUESTS,
        },
        "difficulty": copy.deepcopy(preregistration["difficulty_calibration"]),
        "hardware": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": ENGINE_COUNT,
            "tensor_parallel_per_engine": 1,
            "all_four_engines_receive_complete_union_every_phase": True,
            "prelaunch_and_final_all_four_gpu_idle_required": True,
        },
        "persistence": {
            "provisional_pool_fields": [
                "row_sha256", "panel", "stratum", "calibration_rank",
            ],
            "raw_content_or_numeric_values_persisted": False,
            "dense_value_vectors_persisted": False,
            "dense_value_vector_commitments_only": True,
        },
        "future_manual_review_shuffle": {
            "algorithm": (
                "ascending_sha256(utf8(domain)||utf8(prereg_content_sha256)"
                "||utf8(row_sha256))"
            ),
            "domain": "specialist-v35a-blinded-manual-review-order-v1",
            "content_independent": True,
            "reviewer_receives_rank_or_numeric_value": False,
        },
        "authority": {
            "manual_review_of_provisional_training_rows": True,
            "adopt_hard_tier": False,
            "run_replay_HPO": False,
            "model_update": False,
            "checkpoint_write": False,
            "evaluation_open": False,
            "dataset_promotion": False,
        },
        "worker_extension": WORKER_EXTENSION,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    return _seal(value)


def _is_allowed_untracked(relative):
    if relative in ALLOWED_UNTRACKED_PATHS:
        return True
    if any(relative.startswith(prefix) for prefix in ALLOWED_UNTRACKED_PREFIXES):
        return True
    match = CURATOR_UNTRACKED_PATTERN.fullmatch(relative)
    return match is not None and int(match.group(1)) >= 390


def validate_worktree_status(raw_status):
    if not isinstance(raw_status, str):
        raise TypeError("V35A worktree status must be text")
    allowed = []
    for raw_line in raw_status.splitlines():
        if len(raw_line) < 4:
            raise RuntimeError("V35A worktree status record changed")
        status, relative = raw_line[:2], raw_line[3:]
        if status != "??" or not _is_allowed_untracked(relative):
            raise RuntimeError(
                "V35A real launch requires committed-clean source outside the allowlist"
            )
        allowed.append(relative)
    return {
        "all_tracked_files_clean": True,
        "only_explicitly_allowlisted_untracked_paths_present": True,
        "allowed_untracked_entry_count": len(allowed),
        "allowed_untracked_paths_sha256": canonical_sha256(sorted(allowed)),
    }


def certify_committed_source(implementation, expected_source_commit):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if head != expected_source_commit or len(head) != 40:
        raise RuntimeError("V35A expected source commit is not current HEAD")
    committed = {}
    for name, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"V35A real launch requires committed source: {relative}"
            ) from error
        if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
            raise RuntimeError(f"V35A source differs from commit: {relative}")
        committed[name] = item["file_sha256"]
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, text=True,
    )
    return _seal({
        "schema": "eggroll-es-v35a-committed-clean-source-certificate",
        "git_head": head,
        "committed_file_count": len(committed),
        "committed_files_sha256": canonical_sha256(committed),
        **validate_worktree_status(status),
    })


def _request_identity(prompt_items):
    return canonical_sha256([item["prompt_token_ids"] for item in prompt_items])


def _score_union_outputs(dense_items, outputs):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    values = np.asarray([
        item["mean_answer_token_logprob"] for item in dense["examples"]
    ], dtype=np.float64)
    if values.shape != (UNION_ROWS,) or not np.isfinite(values).all():
        raise RuntimeError("V35A dense score coverage changed")
    token_traces = []
    for item, output in zip(dense_items, outputs):
        returned_ids = list(getattr(output, "prompt_token_ids", None) or [])
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if returned_ids != item["prompt_token_ids"] or prompt_logprobs is None:
            raise RuntimeError("V35A exact prompt-token trace changed")
        trace = []
        start = item["answer_token_start"]
        stop = start + item["answer_token_count"]
        for position in range(start, stop):
            token_id = returned_ids[position]
            candidates = prompt_logprobs[position]
            if candidates is None or token_id not in candidates:
                raise RuntimeError("V35A exact gold-token logprob is missing")
            selected = candidates[token_id]
            logprob = float(
                selected.logprob
                if hasattr(selected, "logprob") else selected["logprob"]
            )
            if not math.isfinite(logprob):
                raise RuntimeError("V35A exact gold-token logprob is non-finite")
            trace.append([token_id, logprob])
        token_traces.append(trace)
    return values, canonical_sha256({
        "dense_result": dense,
        "exact_gold_token_id_and_logprob_traces": token_traces,
    })


def phase_identity(phase):
    values, commitments = phase
    values = np.asarray(values, dtype=np.float64)
    if (
        values.shape != (ENGINE_COUNT, UNION_ROWS)
        or not np.isfinite(values).all()
        or len(commitments) != ENGINE_COUNT
    ):
        raise RuntimeError("V35A phase geometry changed")
    return {
        "value_matrix_shape": [ENGINE_COUNT, UNION_ROWS],
        "value_matrix_byte_sha256": hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest(),
        "dense_commitments_sha256": canonical_sha256(commitments),
    }


def assert_phase_exact(phase):
    values, commitments = phase
    identity = phase_identity(phase)
    if (
        not all(np.array_equal(values[0], values[index]) for index in range(1, 4))
        or len(set(commitments)) != 1
    ):
        raise RuntimeError("V35A cross-engine dense outputs are not exact")
    return identity


def assert_cross_phase_exact(phases):
    if list(phases) != list(PHASES):
        raise RuntimeError("V35A phase order changed")
    reference_values, reference_commitments = phases["A"]
    if any(
        not np.array_equal(reference_values, phases[name][0])
        or reference_commitments != phases[name][1]
        for name in ("B", "C")
    ):
        raise RuntimeError("V35A A/B/C dense results are not exact")
    identities = {name: assert_phase_exact(phases[name]) for name in PHASES}
    return _seal({
        "schema": "eggroll-es-v35a-A-B-C-exactness-audit",
        "phase_identity_sha256": canonical_sha256(identities),
        "all_four_engines_exact_within_each_phase": True,
        "all_dense_results_exact_across_A_B_C": True,
        "phase_B_only_calibration_estimand": True,
    })


def build_provisional_pool(metadata, phase_b_values, preregistration):
    values = np.asarray(phase_b_values, dtype=np.float64)
    if values.shape != (ENGINE_COUNT, UNION_ROWS) or not all(
        np.array_equal(values[0], values[index]) for index in range(1, 4)
    ):
        raise RuntimeError("V35A ranking requires exact four-engine phase B")
    grouped = defaultdict(list)
    for index, item in enumerate(metadata):
        grouped[(item["panel"], item["stratum"])].append(
            (float(values[0, index]), item["row_sha256"])
        )
    expected = preregistration["difficulty_calibration"][
        "tier_counts_per_panel"
    ]
    provisional = []
    ranked_identity = {}
    for panel in OPTIMIZATION_PANELS:
        for stratum in STRATA:
            rows = sorted(grouped[(panel, stratum)], key=lambda pair: pair)
            panel_rows = expected[stratum]["panel_rows"]
            take = expected[stratum]["provisional_candidates"]
            if len(rows) != panel_rows or take != math.ceil(0.5 * panel_rows):
                raise RuntimeError("V35A preregistered stratum allocation changed")
            ranked_identity[f"{panel}:{stratum}"] = canonical_sha256([
                row_sha for _value, row_sha in rows
            ])
            for rank, (_value, row_sha) in enumerate(rows[:take], start=1):
                provisional.append({
                    "row_sha256": row_sha,
                    "panel": panel,
                    "stratum": stratum,
                    "calibration_rank": rank,
                })
    if (
        len(provisional) != 87
        or Counter(item["panel"] for item in provisional)
        != Counter({name: 29 for name in OPTIMIZATION_PANELS})
    ):
        raise RuntimeError("V35A provisional review allocation changed")
    return provisional, canonical_sha256(ranked_identity)


def deterministic_manual_review_order(provisional):
    domain = b"specialist-v35a-blinded-manual-review-order-v1"
    prefix = domain + PREREG_CONTENT_SHA256.encode("ascii")
    ordered = sorted(
        provisional,
        key=lambda item: hashlib.sha256(
            prefix + item["row_sha256"].encode("ascii")
        ).hexdigest(),
    )
    if len(ordered) != 87 or len({item["row_sha256"] for item in ordered}) != 87:
        raise RuntimeError("V35A manual-review shuffle coverage changed")
    return [item["row_sha256"] for item in ordered]


class LaggedReplayCalibrationRuntimeMixinV35A:
    def configure_lagged_replay_calibration_v35a(
        self, preregistration, rows, layer_bundle,
    ):
        if (
            len(self.engines) != ENGINE_COUNT
            or self.n_vllm_engines != ENGINE_COUNT
            or self.n_gpu_per_vllm_engine != 1
            or not math.isclose(float(self.sigma), 0.0, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RuntimeError("V35A exact four-TP1 unperturbed recipe changed")
        devices = [
            runtime_v23a._unwrap_one_v23a(item, "runtime_device_identity_v23a")
            for item in self._resolve([
                engine.collective_rpc.remote(
                    "runtime_device_identity_v23a", args=(f"calibration_slot_{rank}",)
                )
                for rank, engine in enumerate(self.engines)
            ])
        ]
        if any(
            item.get("rank") != rank
            or item.get("world_size") != ENGINE_COUNT
            or item.get("arm") != f"calibration_slot_{rank}"
            or item.get("cuda_visible_devices") != str(rank)
            or item.get("runtime_cuda_device") != 0
            or item.get("update_surfaces_closed") is not True
            for rank, item in enumerate(devices)
        ) or len({item["cuda_visible_devices"] for item in devices}) != 4:
            raise RuntimeError("V35A fixed rank-to-physical-GPU identity changed")
        installs = [
            runtime_v23a._unwrap_one_v23a(item, "install_layer_plan_v4")
            for item in self._resolve([
                engine.collective_rpc.remote(
                    "install_layer_plan_v4",
                    args=(
                        Path(layer_bundle["path"]).read_bytes(),
                        layer_bundle["file_sha256"],
                        layer_bundle["plan_sha256"],
                        anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                        anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
                    ),
                )
                for engine in self.engines
            ])
        ]
        if any(
            item.get("installed") is not True
            or item.get("rank") != rank
            or item.get("world_size") != ENGINE_COUNT
            or item.get("plan_sha256", item.get("layer_plan_sha256"))
            != layer_bundle["plan_sha256"]
            or item.get("runtime_selected_parameter_count") != 23
            or item.get("selected_element_count") != 142_999_552
            for rank, item in enumerate(installs)
        ):
            raise RuntimeError("V35A layer-plan installation changed")
        self._v35a_rows = copy.deepcopy(rows)
        self._v35a_preregistration = copy.deepcopy(preregistration)
        return _seal({
            "schema": "eggroll-es-v35a-runtime-configuration",
            "device_identity_sha256": canonical_sha256(devices),
            "installation_sha256": canonical_sha256(installs),
            "four_distinct_physical_gpu_assignments": True,
            "update_checkpoint_evaluation_surfaces_closed": True,
        })

    def _prepare_union_v35a(self):
        prompts = [
            runtime_v23a.base.specialist_template(item["question"])
            for item in self._v35a_rows
        ]
        answers = [item["answer"] for item in self._v35a_rows]
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, prompts, answers
        )
        prompt_items = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in dense_items
        ]
        if (
            len(dense_items) != UNION_ROWS
            or not all(len(item["prompt_token_ids"]) <= 1024 for item in dense_items)
        ):
            raise RuntimeError("V35A fixed dense token contract changed")
        return dense_items, prompt_items, _seal({
            "schema": "eggroll-es-v35a-token-request-audit",
            "request_count": UNION_ROWS,
            "request_identity_sha256": _request_identity(prompt_items),
            "dense_token_contract_sha256": canonical_sha256(dense_items),
            "all_within_frozen_1024_token_cap": True,
        })

    def _score_phase_v35a(self, dense_items, prompt_items):
        batches = self._resolve([
            engine.generate.remote(
                list(prompt_items), self._dense_sampling_params_v4(0), use_tqdm=False,
            )
            for engine in self.engines
        ])
        if len(batches) != ENGINE_COUNT or any(
            len(batch) != UNION_ROWS for batch in batches
        ):
            raise RuntimeError("V35A all-engine generation coverage changed")
        values = np.empty((ENGINE_COUNT, UNION_ROWS), dtype=np.float64)
        commitments = []
        for engine_index, outputs in enumerate(batches):
            values[engine_index], digest = _score_union_outputs(dense_items, outputs)
            commitments.append(digest)
        phase = (values, commitments)
        assert_phase_exact(phase)
        return phase

    def capture_calibration_v35a(self):
        dense_items, prompt_items, token_audit = self._prepare_union_v35a()
        phases = {}
        activity = {}
        for name in PHASES:
            phases[name] = self._score_phase_v35a(dense_items, prompt_items)
            observation = runtime_v33a._observe_all_four_gpus_v33a()
            runtime_v33a._physical_gpu_identity_v33a(observation)
            if any(
                item["running_process_count"] < 1 for item in observation["gpus"]
            ):
                raise RuntimeError("V35A did not observe activity on every GPU")
            activity[name] = canonical_sha256(observation)
        exactness = assert_cross_phase_exact(phases)
        metadata = [
            {
                "panel": item["panel"],
                "stratum": item["stratum"],
                "row_sha256": item["row_sha256"],
            }
            for item in self._v35a_rows
        ]
        phase_commitments = {
            name: phase_identity(phases[name]) for name in PHASES
        }
        audit = _seal({
            "schema": "eggroll-es-v35a-content-free-runtime-audit",
            "token_request_audit_sha256": token_audit[
                "content_sha256_before_self_field"
            ],
            "exactness_audit": exactness,
            "phase_dense_value_commitments_sha256": canonical_sha256(
                phase_commitments
            ),
            "all_four_gpu_activity_each_phase_sha256": canonical_sha256(activity),
            "all_four_gpu_activity_each_phase": True,
            "calibration_estimand_phase": "B",
            "phase_A_or_C_used_for_ranking": False,
            "calibration_estimand_requests": REQUESTS_PER_PHASE,
            "integrity_repeat_requests": 2 * REQUESTS_PER_PHASE,
            "physical_total_requests": PHYSICAL_REQUESTS,
            "raw_content_or_dense_values_persisted": False,
        })
        assert_content_free(audit)
        return np.copy(phases["B"][0]), metadata, audit

    @staticmethod
    def _closed_surface_v35a(*_args, **_kwargs):
        raise RuntimeError(
            "V35A closes update checkpoint evaluation promotion and replay surfaces"
        )

    configure_anchor = _closed_surface_v35a
    configure_train_panels_v13 = _closed_surface_v35a
    estimate_train_panels_v13 = _closed_surface_v35a
    estimate_step_coefficients = _closed_surface_v35a
    apply_seed_coefficients = _closed_surface_v35a
    train_step = _closed_surface_v35a
    evaluate_handle = _closed_surface_v35a
    evaluate_population_on_batch = _closed_surface_v35a
    eval_step = _closed_surface_v35a
    fit = _closed_surface_v35a


def load_runtime_trainer(preregistration, layer_bundle):
    model = preregistration["frozen_recipe"]["model"]
    parent = anchor_v4.load_trainer(layer_bundle)

    class LaggedReplayCalibrationRuntimeTrainerV35A(
        LaggedReplayCalibrationRuntimeMixinV35A, parent,
    ):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("V35A requires exactly four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            groups = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}], strategy="PACK"
                )
                for _ in range(4)
            ]
            self._v35a_partial_groups = groups
            ray.get([group.ready() for group in groups])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for group in groups
            ]
            engines = []
            self._v35a_partial_engines = engines
            for rank in range(4):
                engines.append(ray.remote(
                    num_cpus=0, num_gpus=1, scheduling_strategy=strategies[rank]
                )(ESNcclLLM).remote(
                    model=model,
                    tensor_parallel_size=1,
                    worker_extension_cls=WORKER_EXTENSION,
                    dtype=precision,
                    enable_prefix_caching=False,
                    enforce_eager=True,
                    gpu_memory_utilization=0.82,
                    max_model_len=2048,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    mm_processor_cache_gb=0,
                    skip_mm_profiling=True,
                    moe_backend="triton",
                ))
            return engines, groups

    return LaggedReplayCalibrationRuntimeTrainerV35A


def make_trainer_fail_closed(preregistration, layer_bundle):
    """Keep a cleanup handle even when the trainer constructor raises."""
    cls = load_runtime_trainer(preregistration, layer_bundle)
    trainer = cls.__new__(cls)
    try:
        cls.__init__(
            trainer,
            model_name=preregistration["frozen_recipe"]["model"],
            checkpoint=None,
            sigma=0.0,
            alpha=0.0,
            population_size=4,
            reward_shaping="z-scores",
            num_iterations=1,
            max_tokens=1,
            batch_size=UNION_ROWS,
            mini_batch_size=UNION_ROWS,
            reward_function=runtime_v23a.base.specialist_reward,
            template_function=runtime_v23a.base.specialist_template,
            train_dataloader=[],
            eval_dataloader_dict={},
            eval_freq=1,
            n_vllm_engines=4,
            n_gpu_per_vllm_engine=1,
            logging="none",
            global_seed=43,
            use_gpus="0,1,2,3",
            experiment_name=EXPERIMENT_NAME,
            wandb_project="specialist-eggroll-es",
            save_best_models=False,
            reward_function_timeout=10,
            output_directory=str(OUTPUT_DIRECTORY),
        )
    except BaseException as error:
        setattr(error, "_v35a_partial_trainer", trainer)
        raise
    return trainer


def close_trainer_fail_closed(trainer):
    failure = None
    base_cleanup_succeeded = False
    try:
        runtime_v23a.base.close_trainer(trainer)
        base_cleanup_succeeded = True
    except BaseException as error:
        failure = error
    try:
        import ray
        from ray.util import remove_placement_group

        engines = list(getattr(trainer, "_v35a_partial_engines", []))
        for engine in engines:
            try:
                ray.kill(engine, no_restart=True)
            except BaseException:
                pass
        groups = list(getattr(trainer, "_v35a_partial_groups", []))
        for group in groups:
            try:
                remove_placement_group(group)
            except BaseException:
                pass
    except (ImportError, AttributeError):
        pass
    pool = getattr(trainer, "mp_pool", None)
    if pool is not None and not base_cleanup_succeeded:
        try:
            pool.close()
            pool.join()
        except BaseException as error:
            if failure is None:
                failure = error
    try:
        import ray
        ray.shutdown()
    except (ImportError, AttributeError):
        pass
    if failure is not None:
        raise failure


def recheck_postcleanup_bindings(
    preregistration, manifest, layer_bundle, implementation, recipe,
    committed_source,
):
    current_preregistration = load_preregistration()
    current_manifest = load_manifest(current_preregistration)
    current_layer = load_layer_bundle(current_preregistration)
    if file_sha256(Path(manifest["source"]["path"])) != SOURCE_FILE_SHA256:
        raise RuntimeError("V35A post-cleanup training source binding changed")
    if current_preregistration != preregistration or current_manifest != manifest:
        raise RuntimeError("V35A post-cleanup frozen input changed")
    if current_layer["plan_sha256"] != layer_bundle["plan_sha256"]:
        raise RuntimeError("V35A post-cleanup layer binding changed")
    if implementation_identity()["bundle_sha256"] != implementation["bundle_sha256"]:
        raise RuntimeError("V35A post-cleanup implementation changed")
    recertified = certify_committed_source(
        implementation, committed_source["git_head"]
    )
    if recertified["git_head"] != committed_source["git_head"]:
        raise RuntimeError("V35A post-cleanup source commit changed")
    if recipe_v35a(
        current_preregistration, current_manifest, current_layer, implementation
    )["content_sha256_before_self_field"] != recipe["content_sha256_before_self_field"]:
        raise RuntimeError("V35A post-cleanup recipe changed")
    validate_live_model(current_preregistration)
    return _seal({
        "schema": "eggroll-es-v35a-postcleanup-binding-recheck",
        "all_frozen_bindings_unchanged": True,
        "performed_after_gpu_cleanup_before_report": True,
    })


def _write_attempt(value):
    assert_content_free(value)
    runtime_v23a._exclusive_write_json_v23a(ATTEMPT_PATH, value)


def _rewrite_attempt(value):
    assert_content_free(value)
    runtime_v23a._rewrite_json_v23a(ATTEMPT_PATH, value)


def run_exact_v35a(
    preregistration, manifest, rows, source_projection_audit, layer_bundle,
    implementation, recipe, committed_source,
):
    if ATTEMPT_PATH.exists() or RUN_DIRECTORY.exists():
        raise RuntimeError("V35A requires fresh exclusive attempt and run paths")
    environment = runtime_r2.certify_runtime_environment_r2()
    live_model = validate_live_model(preregistration)
    prelaunch_idle = runtime_v33a.assert_all_four_gpus_idle_v33a()
    attempt = _seal({
        "schema": "eggroll-es-v35a-durable-launch-attempt",
        "status": "launching",
        "phase": "before_run_directory_reservation",
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "committed_source_certificate_sha256": committed_source[
            "content_sha256_before_self_field"
        ],
        "runtime_environment_certificate_sha256": environment[
            "content_sha256_before_self_field"
        ],
        "live_model_audit_sha256": live_model["content_sha256_before_self_field"],
        "prelaunch_idle_certificate_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "final_idle_certificate_sha256": None,
        "report_binding": None,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "dataset_promotion_applied": False,
        "replay_HPO_run": False,
    })
    _write_attempt(attempt)
    try:
        RUN_DIRECTORY.mkdir(mode=0o755, parents=False, exist_ok=False)
    except BaseException as error:
        final_idle = runtime_v33a.wait_for_final_gpu_idle_v33a(prelaunch_idle)
        attempt.update({
            "status": "failed",
            "phase": "run_directory_reservation",
            "failure_type": type(error).__name__,
            "failure_sha256": canonical_sha256(type(error).__name__),
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        raise
    trainer = None
    phase_b_values = metadata = runtime_audit = None
    failure = None
    final_idle = None
    configuration = None
    try:
        runtime_v23a.base.set_seed(43)
        try:
            trainer = make_trainer_fail_closed(preregistration, layer_bundle)
        except BaseException as constructor_error:
            trainer = getattr(constructor_error, "_v35a_partial_trainer", None)
            raise
        configuration = trainer.configure_lagged_replay_calibration_v35a(
            preregistration, rows, layer_bundle
        )
        phase_b_values, metadata, runtime_audit = (
            trainer.capture_calibration_v35a()
        )
    except BaseException as error:
        failure = error
    finally:
        if trainer is not None:
            try:
                close_trainer_fail_closed(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
        try:
            final_idle = runtime_v33a.wait_for_final_gpu_idle_v33a(prelaunch_idle)
        except BaseException as idle_error:
            if failure is None:
                failure = idle_error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "after_fail_closed_cleanup",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "phase": "gpu_capture_or_cleanup",
            }),
            "final_idle_certificate_sha256": (
                None if final_idle is None
                else final_idle["content_sha256_before_self_field"]
            ),
            "all_four_gpus_idle_after_cleanup": final_idle is not None,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        raise failure
    try:
        postcleanup = recheck_postcleanup_bindings(
            preregistration, manifest, layer_bundle, implementation, recipe,
            committed_source,
        )
        provisional, ranked_identity_sha256 = build_provisional_pool(
            metadata, phase_b_values, preregistration
        )
        review_order_sha256 = canonical_sha256(
            deterministic_manual_review_order(provisional)
        )
        phase_b_values = None
        metadata = None
        report = _seal({
            "schema": "eggroll-es-lagged-replay-calibration-report-v35a",
            "status": "completed_calibration_pending_blinded_manual_review",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "source_projection_audit_sha256": source_projection_audit[
                "content_sha256_before_self_field"
            ],
            "configuration": configuration,
            "runtime_audit": runtime_audit,
            "ranked_identity_sha256": ranked_identity_sha256,
            "deterministic_blinded_review_order_sha256": review_order_sha256,
            "provisional_review_pool": provisional,
            "provisional_review_pool_count": len(provisional),
            "postcleanup_binding_recheck_sha256": postcleanup[
                "content_sha256_before_self_field"
            ],
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
            "hard_tier_adopted": False,
            "manual_review_complete": False,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
            "replay_HPO_run": False,
            "direct_action_taken": False,
        })
        assert_content_free(report)
        runtime_v23a._exclusive_write_json_v23a(REPORT_PATH, report)
        attempt.update({
            "status": "complete",
            "phase": "after_compact_report_and_final_gpu_cleanup",
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "report_binding": {
                "relative_path": REPORT_PATH.relative_to(ROOT).as_posix(),
                "file_sha256": file_sha256(REPORT_PATH),
                "content_sha256": report["content_sha256_before_self_field"],
            },
            "all_four_gpus_idle_after_cleanup": True,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        return report
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "after_cleanup_during_content_free_report_finalization",
            "failure_type": type(error).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(error).__name__, "phase": "report_finalization",
            }),
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        raise


def validate_runtime_args(args, preregistration, implementation, recipe):
    if any(os.environ.get(key) for key in runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A):
        raise ValueError("V35A rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("V35A rejects the vLLM batch-invariant backend swap")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256,
         implementation["bundle_sha256"], "implementation bundle"),
        (args.expected_recipe_sha256,
         recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if expected is not None and expected != actual:
            raise ValueError(f"V35A expected {label} hash changed")
    if args.v35a_dry_run:
        return None
    if args.expected_implementation_bundle_sha256 is None:
        raise ValueError("V35A real launch requires exact expected implementation bundle")
    if args.expected_recipe_sha256 is None:
        raise ValueError("V35A real launch requires exact expected recipe")
    if args.expected_source_commit is None:
        raise ValueError("V35A real launch requires exact expected source commit")
    return certify_committed_source(implementation, args.expected_source_commit)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v35a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    parser.add_argument("--expected-source-commit")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    assert_train_only_argv(argv)
    args = _parser().parse_args(argv)
    preregistration = load_preregistration()
    manifest = load_manifest(preregistration)
    rows, source_projection_audit = materialize_optimization_union(
        preregistration, manifest
    )
    layer_bundle = load_layer_bundle(preregistration)
    implementation = implementation_identity()
    recipe = recipe_v35a(preregistration, manifest, layer_bundle, implementation)
    committed_source = validate_runtime_args(
        args, preregistration, implementation, recipe
    )
    if args.v35a_dry_run:
        payload = _seal({
            "schema": "eggroll-es-lagged-replay-calibration-dry-run-v35a",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "source_projection_audit_sha256": source_projection_audit[
                "content_sha256_before_self_field"
            ],
            "request_accounting": copy.deepcopy(recipe["request_accounting"]),
            "provisional_candidate_count": 87,
            "final_hard_tier_capacity_after_manual_review": 39,
            "phase_B_only_calibration_estimand": True,
            "screen_panels_untouched": True,
            "fresh_exclusive_attempt_and_run_directory_required": True,
            "real_launch_requires_exact_committed_implementation_recipe_and_HEAD": True,
            "content_free_persistence_only": True,
            "gpu_launched": False,
            "runtime_launched": False,
            "model_update_checkpoint_evaluation_promotion_or_replay_HPO_authorized": False,
        })
        assert_content_free(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v35a(
        preregistration, manifest, rows, source_projection_audit, layer_bundle,
        implementation, recipe, committed_source,
    )


if __name__ == "__main__":
    main()
