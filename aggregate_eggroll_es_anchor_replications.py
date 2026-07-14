#!/usr/bin/env python3
"""Fail-closed aggregation for corrected anchored EGGROLL line searches.

This consumer deliberately reads only completed line-search journals.  It
never opens evaluation result files and never accepts a heldout split or path.
Exploratory monotonic alpha searches and direct ``0 -> alpha`` confirmations
are different report modes because incremental BF16 updates are path
dependent.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path


ALLOWED_SCHEMAS = {
    "eggroll-es-anchor-alpha-line-search-v2",
    "eggroll-es-anchor-alpha-line-search-v3",
    "eggroll-es-anchor-alpha-line-search-v4",
}
ALLOWED_EVAL_SPLITS = ("validation", "ood_qa")
DEFAULT_CONFIRMATION_SEEDS = (42, 43, 44, 45, 46)
RISK_PENALTY = 0.5
MIN_POSITIVE_SEEDS = 4
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FLOAT_TOLERANCE = 1e-12
V3_ENGINE_COUNT = 4
V3_SNAPSHOT_IMPLEMENTATION_KEYS = {
    "distributed_driver_v3",
    "distributed_trainer_v3",
    "distributed_worker_v3",
}
V3_DISTRIBUTED_POLICY = {
    "engine_count": V3_ENGINE_COUNT,
    "tp_per_engine": 1,
    "seed_sharding": "strided_by_inter_engine_rank",
    "collective_dtype": "torch.float32",
    "two_phase_commit": True,
    "final_hash_consensus_required": True,
    "reference_recapture_policy": "once_before_next_population_only",
    "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
    "direct_alpha_confirmation_required": True,
}
V4_SNAPSHOT_IMPLEMENTATION_KEYS = {
    "distributed_driver_v4",
    "distributed_trainer_v4",
    "distributed_worker_v4",
}
V4_MODEL_CONFIG_SHA256 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
V4_FROZEN_LAYER_PLANS = {
    "6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb": {
        "file_sha256": (
            "8e855cbd0d6130278e87b1af348e39dd0f683b8575d9abcb9260f3fe7b29d824"
        ),
        "checkpoint_to_runtime_mapping_sha256": (
            "0a1b84e8ed53ef56c174e7fcac728a4820293505647ab6b9ea02bc86a012b3b1"
        ),
        "runtime_selected_name_sha256": (
            "417b3867ba9a56f909d01b1e7bb0b8bb04f903c3ec49438a6675239a7bab270f"
        ),
    },
    "b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0": {
        "file_sha256": (
            "f2b38054e3cdaf41619cce579d3ba2e030fa3cfa87fd42b50543f655ff5f6dc0"
        ),
        "checkpoint_to_runtime_mapping_sha256": (
            "d6f43de81bb5c41318a38f077b8a3e6272676801752ff68d4772977ac72182f7"
        ),
        "runtime_selected_name_sha256": (
            "a7df9257f81c05a3fb3e858209486bd930aad0ddb94d7398e1644b779fb8b70d"
        ),
    },
}
V4_SOURCE_UNIT_COUNT = 70
V4_RUNTIME_PARAMETER_COUNT = 46
V4_SELECTED_ELEMENT_COUNT = 285_999_104
V4_SELECTED_BYTE_COUNT = 571_998_208
V4_BINDING_KEYS = {
    "layer_plan_file_sha256",
    "layer_plan_sha256",
    "checkpoint_to_runtime_mapping_sha256",
    "source_unit_count",
    "runtime_selected_name_sha256",
    "selected_parameter_manifest_sha256",
    "runtime_selected_parameter_count",
    "selected_element_count",
    "unselected_origin_sha256",
    "dense_reward_sha256",
}
V4_DENSE_REWARD_CONFIG = {
    "schema": "eggroll-es-dense-qa-reward-v1",
    "objective": "teacher_forced_gold_answer_prompt_logprob",
    "text_construction": "exact_prompt_plus_answer",
    "tokenization": {
        "add_special_tokens": False,
        "append_eos": False,
        "max_total_tokens": 1024,
        "require_prompt_token_prefix": True,
        "truncation": False,
    },
    "scored_positions": "answer_tokens_only",
    "aggregation": "mean_tokens_per_example_then_mean_examples",
    "generation": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "prompt_logprobs": 1,
        "detokenize": False,
    },
}
V4_DENSE_REWARD_SHA256 = (
    "4941f2e94091b1f8e7ab7b5294ebc6520b80aba1326b7dc6ccea5140a3da5da2"
)
V4_FROZEN_EVAL_BATCH_SIZE = 64
V4_FROZEN_S6_SPLITS = {
    "train": {
        "rows": 794,
        "sha256": (
            "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"
        ),
    },
    "validation": {
        "rows": 41,
        "sha256": (
            "19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db"
        ),
    },
    "ood_qa": {
        "rows": 24,
        "sha256": (
            "b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced"
        ),
    },
}
V4_FROZEN_S6_ANCHOR = {
    "rows": 128,
    "sha256": "a693e23c48e558e9b72c30b0ae31f0b3e580a665371846978ad4d3eca7ef5f7d",
    "report_sha256": (
        "913ff2cb786ac50ffe86770291b6173a14220afce3682dfea67359c45cf6e9f5"
    ),
}
V4_FROZEN_S6_BASELINE = {
    "validation": {
        "rows": 41, "exact": 2, "nonzero": 13,
        "mean_reward": 0.08381010452961674,
    },
    "ood_qa": {
        "rows": 24, "exact": 16, "nonzero": 23,
        "mean_reward": 0.714128787878788,
    },
    "ood_prose": {
        "item_count": 16,
        "scored_token_count": 10926,
        "mean_token_logprob": -1.2632580042542214,
    },
}


class JournalValidationError(ValueError):
    """A journal is incomplete, unsafe, inconsistent, or non-comparable."""


def canonical_sha256(value):
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _fail(message):
    raise JournalValidationError(message)


def _require(condition, message):
    if not condition:
        _fail(message)


def _finite_float(value, label):
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    value = float(value)
    _require(math.isfinite(value), f"{label} must be finite")
    return value


def _integer(value, label, minimum=None):
    _require(
        not isinstance(value, bool) and isinstance(value, int),
        f"{label} must be an integer",
    )
    if minimum is not None:
        _require(value >= minimum, f"{label} is below {minimum}")
    return value


def _sha256(value, label):
    _require(
        isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _same_float(left, right):
    try:
        left = float(left)
        right = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(
            left, right, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE,
        )
    )


def _require_exact_fields(value, expected, label):
    _require(isinstance(value, dict), f"{label} is missing")
    for key, expected_value in expected.items():
        actual = value.get(key)
        _require(
            actual == expected_value
            and type(actual) is type(expected_value),
            f"{label} {key} changed",
        )


def _validate_weight_identity(identity, label):
    _require(isinstance(identity, dict), f"{label} is missing")
    _require(
        identity.get("schema") == "eggroll-es-weight-state-sha256-v2",
        f"{label} schema changed",
    )
    return {
        "schema": identity["schema"],
        "sha256": _sha256(identity.get("sha256"), f"{label} SHA-256"),
        "parameter_count": _integer(
            identity.get("parameter_count"),
            f"{label} parameter count", minimum=1,
        ),
        "total_bytes": _integer(
            identity.get("total_bytes"), f"{label} byte count", minimum=1,
        ),
    }


def _validate_v4_bindings(bindings, label):
    _require(isinstance(bindings, dict), f"{label} is missing")
    _require(
        set(bindings) == V4_BINDING_KEYS,
        f"{label} fields changed",
    )
    plan_sha = _sha256(bindings.get("layer_plan_sha256"), f"{label} plan SHA-256")
    frozen = V4_FROZEN_LAYER_PLANS.get(plan_sha)
    _require(frozen is not None, f"{label} plan is not frozen")
    _require(
        bindings.get("layer_plan_file_sha256") == frozen["file_sha256"],
        f"{label} file/plan identity pair changed",
    )
    _require(
        bindings.get("checkpoint_to_runtime_mapping_sha256")
        == frozen["checkpoint_to_runtime_mapping_sha256"],
        f"{label} checkpoint/runtime mapping changed",
    )
    _require(
        bindings.get("runtime_selected_name_sha256")
        == frozen["runtime_selected_name_sha256"],
        f"{label} selected runtime names changed",
    )
    for key in (
        "layer_plan_file_sha256", "selected_parameter_manifest_sha256",
        "unselected_origin_sha256", "dense_reward_sha256",
    ):
        _sha256(bindings.get(key), f"{label} {key}")
    _require(
        bindings.get("source_unit_count") == V4_SOURCE_UNIT_COUNT
        and type(bindings.get("source_unit_count")) is int
        and bindings.get("runtime_selected_parameter_count")
        == V4_RUNTIME_PARAMETER_COUNT
        and type(bindings.get("runtime_selected_parameter_count")) is int
        and bindings.get("selected_element_count") == V4_SELECTED_ELEMENT_COUNT
        and type(bindings.get("selected_element_count")) is int,
        f"{label} frozen selection counts changed",
    )
    _require(
        bindings.get("dense_reward_sha256") == V4_DENSE_REWARD_SHA256,
        f"{label} dense reward identity changed",
    )
    return copy.deepcopy(bindings)


def _validate_v4_dense_reward(value, label):
    _require(isinstance(value, dict), f"{label} is missing")
    _require(
        value.get("config") == V4_DENSE_REWARD_CONFIG,
        f"{label} semantics changed",
    )
    _require(
        canonical_sha256(value["config"]) == V4_DENSE_REWARD_SHA256,
        f"{label} independently reconstructed hash changed",
    )
    _require(
        value.get("reward_config_sha256") == V4_DENSE_REWARD_SHA256,
        f"{label} recorded hash changed",
    )
    return {
        "config": copy.deepcopy(V4_DENSE_REWARD_CONFIG),
        "reward_config_sha256": V4_DENSE_REWARD_SHA256,
    }


def _validate_v4_layer_identity(value, label, *, require_runtime):
    _require(isinstance(value, dict), f"{label} is missing")
    path = value.get("path")
    model_config_path = value.get("model_config_path")
    _require(isinstance(path, str) and path, f"{label} path is empty")
    _require(
        isinstance(model_config_path, str) and model_config_path,
        f"{label} model-config path is empty",
    )
    _assert_no_heldout(path, f"{label}.path")
    _assert_no_heldout(model_config_path, f"{label}.model_config_path")
    plan_sha = _sha256(value.get("plan_sha256"), f"{label} plan SHA-256")
    frozen = V4_FROZEN_LAYER_PLANS.get(plan_sha)
    _require(frozen is not None, f"{label} plan is not frozen")
    _require(
        value.get("file_sha256") == frozen["file_sha256"],
        f"{label} file/plan identity pair changed",
    )
    _require(
        value.get("model_config_sha256") == V4_MODEL_CONFIG_SHA256,
        f"{label} model-config identity changed",
    )
    normalized = {
        "path": path,
        "file_sha256": frozen["file_sha256"],
        "plan_sha256": plan_sha,
        "model_config_path": model_config_path,
        "model_config_sha256": V4_MODEL_CONFIG_SHA256,
    }
    if require_runtime:
        bindings = _validate_v4_bindings(
            value.get("runtime_mapping"), f"{label} runtime mapping",
        )
        _require(
            value.get("runtime_mapping_sha256") == canonical_sha256(bindings),
            f"{label} runtime mapping hash changed",
        )
        normalized["runtime_mapping"] = bindings
        normalized["runtime_mapping_sha256"] = canonical_sha256(bindings)
    return normalized


def _assert_no_heldout(value, trail="journal"):
    """Reject heldout-bearing keys or string values without opening paths."""
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if "heldout" in key_text.casefold():
                _fail(f"heldout field is forbidden at {trail}.{key_text}")
            _assert_no_heldout(item, f"{trail}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_heldout(item, f"{trail}[{index}]")
    elif isinstance(value, str) and "heldout" in value.casefold():
        _fail(f"heldout string or path is forbidden at {trail}")


def _verify_content_hash(journal):
    recorded = _sha256(
        journal.get("content_sha256_before_self_field"),
        "journal content hash",
    )
    payload = copy.deepcopy(journal)
    payload.pop("content_sha256_before_self_field", None)
    _require(
        canonical_sha256(payload) == recorded,
        "journal content hash does not match",
    )


def load_journal(path):
    path = Path(path)
    _assert_no_heldout(str(path), "input path")
    try:
        journal = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise JournalValidationError(
            f"cannot load journal {path}: {error}"
        ) from error
    _require(isinstance(journal, dict), "journal root must be an object")
    _assert_no_heldout(journal)
    _verify_content_hash(journal)
    return journal


def _validate_dataset_split(identity, label):
    _require(isinstance(identity, dict), f"{label} identity is missing")
    rows = _integer(identity.get("rows"), f"{label} rows", minimum=1)
    files = identity.get("arrow_files")
    _require(isinstance(files, list) and files, f"{label} has no Arrow files")
    normalized_files = []
    for index, item in enumerate(files):
        _require(isinstance(item, dict), f"{label} Arrow entry is invalid")
        path = item.get("path")
        _require(isinstance(path, str) and path, f"{label} Arrow path is empty")
        _assert_no_heldout(path, f"{label}.arrow_files[{index}].path")
        normalized_files.append({
            "path": path,
            "sha256": _sha256(
                item.get("sha256"), f"{label} Arrow SHA-256",
            ),
        })
    return {"rows": rows, "arrow_files": normalized_files}


def _validate_snapshot(snapshot):
    _require(isinstance(snapshot, dict), "snapshot is missing")
    schema = snapshot.get("schema")
    _require(
        schema in {
            "eggroll-es-anchor-line-search-snapshot-v2",
            "eggroll-es-anchor-line-search-snapshot-v3",
            "eggroll-es-anchor-line-search-snapshot-v4",
        },
        "snapshot schema is not corrected anchored v2/v3/v4",
    )
    train = _validate_dataset_split(snapshot.get("train"), "training")
    evaluations = snapshot.get("evaluations")
    _require(isinstance(evaluations, dict), "evaluation identities are missing")
    _require(
        len(evaluations) == len(ALLOWED_EVAL_SPLITS)
        and set(evaluations) == set(ALLOWED_EVAL_SPLITS),
        "evaluation identities must be exactly validation,ood_qa",
    )
    normalized_evaluations = {
        split: _validate_dataset_split(evaluations[split], split)
        for split in ALLOWED_EVAL_SPLITS
    }

    anchor = snapshot.get("anchor")
    _require(isinstance(anchor, dict), "anchor identity is missing")
    anchor_rows = _integer(anchor.get("rows"), "anchor rows", minimum=1)
    anchor_path = anchor.get("path")
    _require(isinstance(anchor_path, str) and anchor_path, "anchor path is empty")
    _assert_no_heldout(anchor_path, "anchor.path")
    report = anchor.get("report")
    _require(isinstance(report, dict), "anchor report identity is missing")
    report_path = report.get("path")
    _require(
        isinstance(report_path, str) and report_path,
        "anchor report path is empty",
    )
    _assert_no_heldout(report_path, "anchor.report.path")
    normalized_anchor = {
        "path": anchor_path,
        "sha256": _sha256(anchor.get("sha256"), "anchor SHA-256"),
        "rows": anchor_rows,
        "report": {
            "path": report_path,
            "sha256": _sha256(
                report.get("sha256"), "anchor report SHA-256",
            ),
            "schema": report.get("schema"),
            "protected_artifact_count": _integer(
                report.get("protected_artifact_count"),
                "anchor protected artifact count", minimum=0,
            ),
        },
    }
    _require(
        normalized_anchor["report"]["schema"]
        == "general-prose-anchor-build-v1",
        "anchor report schema changed",
    )

    fixed_batch = snapshot.get("fixed_train_batch")
    _require(isinstance(fixed_batch, dict), "fixed train batch identity is missing")
    normalized_batch = {
        "rows": _integer(
            fixed_batch.get("rows"), "fixed train batch rows", minimum=1,
        ),
        "sha256": _sha256(
            fixed_batch.get("sha256"), "fixed train batch SHA-256",
        ),
    }

    implementation = snapshot.get("implementation")
    _require(
        isinstance(implementation, dict) and implementation,
        "implementation identity is missing",
    )
    required_implementation = {
        "driver", "anchor_trainer", "base_trainer", "projection",
        "upstream_trainer", "upstream_worker", "corrected_driver",
        "exact_worker",
    }
    _require(
        required_implementation.issubset(implementation),
        "implementation identity is incomplete",
    )
    normalized_implementation = copy.deepcopy(implementation)
    for key in required_implementation:
        _sha256(implementation.get(key), f"implementation {key}")
    if schema in {
        "eggroll-es-anchor-line-search-snapshot-v3",
        "eggroll-es-anchor-line-search-snapshot-v4",
    }:
        _require(
            V3_SNAPSHOT_IMPLEMENTATION_KEYS.issubset(implementation),
            "v3 implementation identity is incomplete",
        )
        for key in V3_SNAPSHOT_IMPLEMENTATION_KEYS:
            _sha256(implementation.get(key), f"implementation {key}")
        _require_exact_fields(
            snapshot.get("distributed_update_v3"),
            V3_DISTRIBUTED_POLICY,
            "v3 snapshot distributed policy",
        )
    normalized_v4_layer = None
    normalized_v4_reward = None
    if schema == "eggroll-es-anchor-line-search-snapshot-v4":
        _require(
            V4_SNAPSHOT_IMPLEMENTATION_KEYS.issubset(implementation),
            "v4 implementation identity is incomplete",
        )
        for key in V4_SNAPSHOT_IMPLEMENTATION_KEYS:
            _sha256(implementation.get(key), f"implementation {key}")
        normalized_v4_layer = _validate_v4_layer_identity(
            snapshot.get("frozen_layer_plan_v4"),
            "v4 snapshot frozen layer plan", require_runtime=False,
        )
        normalized_v4_reward = _validate_v4_dense_reward(
            snapshot.get("dense_gold_reward_v4"),
            "v4 snapshot dense reward",
        )
        _require_exact_fields(
            snapshot.get("distributed_update_v4"),
            {
                **V3_DISTRIBUTED_POLICY,
                "layer_plan_sha256": normalized_v4_layer["plan_sha256"],
                "dense_reward_sha256": V4_DENSE_REWARD_SHA256,
            },
            "v4 snapshot distributed policy",
        )
        for split_name, split_identity in (
            ("train", train),
            ("validation", normalized_evaluations["validation"]),
            ("ood_qa", normalized_evaluations["ood_qa"]),
        ):
            frozen = V4_FROZEN_S6_SPLITS[split_name]
            _require(
                split_identity["rows"] == frozen["rows"]
                and len(split_identity["arrow_files"]) == 1
                and split_identity["arrow_files"][0]["sha256"]
                == frozen["sha256"],
                f"v4 {split_name} artifact differs from frozen S6",
            )
        _require(
            normalized_anchor["rows"] == V4_FROZEN_S6_ANCHOR["rows"]
            and normalized_anchor["sha256"]
            == V4_FROZEN_S6_ANCHOR["sha256"]
            and normalized_anchor["report"]["sha256"]
            == V4_FROZEN_S6_ANCHOR["report_sha256"],
            "v4 prose anchor differs from frozen S6",
        )

    recipe = snapshot.get("recipe")
    _require(isinstance(recipe, dict), "recipe identity is missing")
    required_recipe = {
        "model_name", "checkpoint", "sigma", "population_size",
        "batch_size", "mini_batch_size", "max_tokens", "seed",
        "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
    }
    _require(required_recipe.issubset(recipe), "recipe identity is incomplete")
    _require(
        isinstance(recipe["model_name"], str) and recipe["model_name"],
        "model name is missing",
    )
    _assert_no_heldout(recipe["model_name"], "recipe.model_name")
    if recipe["checkpoint"] is not None:
        checkpoint = recipe["checkpoint"]
        _require(isinstance(checkpoint, dict), "checkpoint identity is invalid")
        _assert_no_heldout(checkpoint.get("path"), "recipe.checkpoint.path")
        _sha256(checkpoint.get("sha256"), "checkpoint SHA-256")
    _require(_finite_float(recipe["sigma"], "sigma") > 0.0, "sigma must be positive")
    _integer(recipe["population_size"], "population size", minimum=2)
    _integer(recipe["batch_size"], "batch size", minimum=1)
    _integer(recipe["mini_batch_size"], "mini-batch size", minimum=1)
    _integer(recipe["max_tokens"], "max tokens", minimum=1)
    seed = _integer(recipe["seed"], "recipe seed", minimum=0)
    cosine = _finite_float(recipe["min_anchor_cosine"], "anchor cosine")
    _require(0.0 <= cosine < 1.0, "anchor cosine is outside [0,1)")
    _integer(recipe["anchor_items_per_step"], "anchor items", minimum=1)
    targets = recipe["target_alphas"]
    _require(isinstance(targets, list) and len(targets) >= 2, "recipe targets are invalid")
    targets = [_finite_float(value, "recipe target alpha") for value in targets]
    _require(targets[0] == 0.0, "recipe targets do not start at zero")
    _require(
        all(right > left for left, right in zip(targets, targets[1:])),
        "recipe targets are not strictly increasing",
    )

    normalized_recipe = copy.deepcopy(recipe)
    normalized_recipe["target_alphas"] = targets
    if schema in {
        "eggroll-es-anchor-line-search-snapshot-v3",
        "eggroll-es-anchor-line-search-snapshot-v4",
    }:
        _require(
            recipe["population_size"] % V3_ENGINE_COUNT == 0,
            "v3 population size is not divisible by four engines",
        )
    if schema == "eggroll-es-anchor-line-search-snapshot-v4":
        _require(
            recipe["checkpoint"] is None,
            "v4 S6 family has an unfrozen input checkpoint",
        )
        _require(
            Path(recipe["model_name"]).name == "Qwen3.6-35B-A3B",
            "v4 S6 family model changed",
        )
        _require(
            recipe["sigma"] == 0.0003
            and recipe["mini_batch_size"] == V4_FROZEN_EVAL_BATCH_SIZE
            and recipe["max_tokens"] == 32,
            "v4 S6 frozen recipe changed",
        )
    normalized = {
        "schema": schema,
        "train": train,
        "evaluations": normalized_evaluations,
        "anchor": normalized_anchor,
        "fixed_train_batch": normalized_batch,
        "implementation": normalized_implementation,
        "recipe": normalized_recipe,
        "seed": seed,
        "targets": targets,
    }
    if normalized_v4_layer is not None:
        normalized["frozen_layer_plan_v4"] = normalized_v4_layer
        normalized["dense_gold_reward_v4"] = normalized_v4_reward
    return normalized


def _validate_qa_summary(summary, label):
    _require(isinstance(summary, dict), f"{label} summary is missing")
    path = summary.get("path")
    _require(isinstance(path, str) and path, f"{label} output path is empty")
    _assert_no_heldout(path, f"{label}.path")
    rows = _integer(summary.get("rows"), f"{label} rows", minimum=1)
    exact = _integer(summary.get("exact"), f"{label} exact", minimum=0)
    nonzero = _integer(summary.get("nonzero"), f"{label} nonzero", minimum=0)
    _require(exact <= nonzero <= rows, f"{label} counts are inconsistent")
    return {
        "path": path,
        "sha256": _sha256(summary.get("sha256"), f"{label} SHA-256"),
        "rows": rows,
        "mean_reward": _finite_float(
            summary.get("mean_reward"), f"{label} mean reward",
        ),
        "exact": exact,
        "nonzero": nonzero,
    }


def _qa_deltas(baseline, candidate):
    return {
        "mean_reward": candidate["mean_reward"] - baseline["mean_reward"],
        "exact": candidate["exact"] - baseline["exact"],
        "nonzero": candidate["nonzero"] - baseline["nonzero"],
    }


def _validate_qa_gate(recorded, baseline, candidate, label):
    _require(isinstance(recorded, dict), f"{label} QA gate is missing")
    _require(
        recorded.get("schema") == "eggroll-es-strict-qa-nondegradation-v1",
        f"{label} QA gate schema changed",
    )
    for key in (
        "max_mean_reward_degradation", "max_exact_loss", "max_nonzero_loss",
    ):
        _require(recorded.get(key) == 0 or recorded.get(key) == 0.0,
                 f"{label} QA gate is not strict")
    deltas = _qa_deltas(baseline, candidate)
    recorded_deltas = recorded.get("deltas")
    _require(isinstance(recorded_deltas, dict), f"{label} QA deltas are missing")
    _require(
        _same_float(recorded_deltas.get("mean_reward"), deltas["mean_reward"]),
        f"{label} QA mean delta is inconsistent",
    )
    _require(
        recorded_deltas.get("exact") == deltas["exact"],
        f"{label} QA exact delta is inconsistent",
    )
    _require(
        recorded_deltas.get("nonzero") == deltas["nonzero"],
        f"{label} QA nonzero delta is inconsistent",
    )
    passed = (
        deltas["mean_reward"] >= 0.0
        and deltas["exact"] >= 0
        and deltas["nonzero"] >= 0
    )
    _require(recorded.get("passed") is passed, f"{label} QA gate decision is inconsistent")
    return {"deltas": deltas, "passed": passed}


def _validate_prose_summary(summary, label):
    _require(isinstance(summary, dict), f"{label} prose summary is missing")
    path = summary.get("results_path")
    _require(isinstance(path, str) and path, f"{label} prose path is empty")
    _assert_no_heldout(path, f"{label}.prose.results_path")
    return {
        "results_path": path,
        "results_sha256": _sha256(
            summary.get("results_sha256"), f"{label} prose SHA-256",
        ),
        "item_count": _integer(
            summary.get("item_count"), f"{label} prose items", minimum=1,
        ),
        "scored_token_count": _integer(
            summary.get("scored_token_count"),
            f"{label} prose scored tokens", minimum=1,
        ),
        "mean_token_logprob": _finite_float(
            summary.get("mean_token_logprob"),
            f"{label} prose mean token log-probability",
        ),
    }


def _validate_prose_gate(recorded, baseline, candidate, label):
    _require(isinstance(recorded, dict), f"{label} prose gate is missing")
    _require(
        _finite_float(recorded.get("max_degradation"), f"{label} prose margin")
        == 0.0,
        f"{label} prose gate is not strict",
    )
    delta = candidate["mean_token_logprob"] - baseline["mean_token_logprob"]
    _require(
        _same_float(recorded.get("delta"), delta),
        f"{label} prose point delta is inconsistent",
    )
    interval = recorded.get("paired_document_bootstrap_95_ci")
    _require(
        isinstance(interval, list) and len(interval) == 2,
        f"{label} prose confidence interval is missing",
    )
    interval = [
        _finite_float(interval[0], f"{label} prose lower bound"),
        _finite_float(interval[1], f"{label} prose upper bound"),
    ]
    _require(interval[0] <= interval[1], f"{label} prose interval is reversed")
    bootstrap = recorded.get("bootstrap")
    _require(isinstance(bootstrap, dict), f"{label} prose bootstrap is missing")
    _require(
        bootstrap.get("unit") == "normalized_source_url"
        and bootstrap.get("samples") == 20000
        and bootstrap.get("seed") == 20260714
        and bootstrap.get("percentiles") == [0.025, 0.975],
        f"{label} prose bootstrap policy changed",
    )
    _integer(bootstrap.get("document_count"), f"{label} prose documents", minimum=1)
    passed = delta >= 0.0 and interval[0] >= 0.0
    # Older v2 records defined `passed` from the lower interval only.  A
    # positive lower bound should imply a nonnegative point estimate, but the
    # aggregator nevertheless recomputes and explicitly enforces both.
    _require(
        isinstance(recorded.get("passed"), bool),
        f"{label} prose recorded decision is not Boolean",
    )
    _require(
        recorded["passed"] == (interval[0] >= 0.0),
        f"{label} prose recorded decision is inconsistent",
    )
    return {"delta": delta, "paired_document_bootstrap_95_ci": interval, "passed": passed}


def _leaf_dicts(value):
    if isinstance(value, dict):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaf_dicts(item)


def _nested_schema_reports(value, schema):
    reports = []
    if isinstance(value, dict):
        if value.get("schema") == schema:
            reports.append(value)
        for item in value.values():
            reports.extend(_nested_schema_reports(item, schema))
    elif isinstance(value, (list, tuple)):
        for item in value:
            reports.extend(_nested_schema_reports(item, schema))
    return reports


def _validate_v4_partition_identity(identity, partition, bindings, label):
    _require(isinstance(identity, dict), f"{label} is missing")
    _require(
        identity.get("schema") == "eggroll-es-parameter-partition-sha256-v4"
        and identity.get("partition") == partition,
        f"{label} schema or partition changed",
    )
    normalized = {
        "schema": "eggroll-es-parameter-partition-sha256-v4",
        "partition": partition,
        "sha256": _sha256(identity.get("sha256"), f"{label} SHA-256"),
        "parameter_count": _integer(
            identity.get("parameter_count"), f"{label} parameters", minimum=1,
        ),
        "total_elements": _integer(
            identity.get("total_elements"), f"{label} elements", minimum=1,
        ),
        "total_bytes": _integer(
            identity.get("total_bytes"), f"{label} bytes", minimum=1,
        ),
    }
    for key in (
        "layer_plan_file_sha256", "layer_plan_sha256",
        "checkpoint_to_runtime_mapping_sha256",
        "runtime_selected_name_sha256",
        "selected_parameter_manifest_sha256", "dense_reward_sha256",
    ):
        _require(
            identity.get(key) == bindings[key],
            f"{label} {key} changed",
        )
        normalized[key] = bindings[key]
    if partition == "selected":
        _require(
            normalized["parameter_count"] == V4_RUNTIME_PARAMETER_COUNT
            and normalized["total_elements"] == V4_SELECTED_ELEMENT_COUNT
            and normalized["total_bytes"] == V4_SELECTED_BYTE_COUNT
            and normalized["total_bytes"] == 2 * normalized["total_elements"],
            f"{label} frozen BF16 size changed",
        )
    else:
        _require(
            normalized["sha256"] == bindings["unselected_origin_sha256"],
            f"{label} differs from immutable origin",
        )
    return normalized


def _validate_v4_weight_identity(identity, bindings, label):
    _require(isinstance(identity, dict), f"{label} is missing")
    _require(
        identity.get("schema") == "eggroll-es-partitioned-weight-state-v4",
        f"{label} schema changed",
    )
    for key in (
        "layer_plan_file_sha256", "layer_plan_sha256",
        "checkpoint_to_runtime_mapping_sha256", "source_unit_count",
        "runtime_selected_name_sha256",
        "selected_parameter_manifest_sha256", "dense_reward_sha256",
    ):
        _require(identity.get(key) == bindings[key], f"{label} {key} changed")
    selected = _validate_v4_partition_identity(
        identity.get("selected"), "selected", bindings, f"{label} selected",
    )
    unselected = _validate_v4_partition_identity(
        identity.get("unselected"), "unselected", bindings,
        f"{label} unselected",
    )
    payload = {
        "schema": "eggroll-es-partitioned-weight-payload-v4",
        "layer_plan_file_sha256": bindings["layer_plan_file_sha256"],
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "checkpoint_to_runtime_mapping_sha256": (
            bindings["checkpoint_to_runtime_mapping_sha256"]
        ),
        "source_unit_count": bindings["source_unit_count"],
        "runtime_selected_name_sha256": (
            bindings["runtime_selected_name_sha256"]
        ),
        "selected_parameter_manifest_sha256": (
            bindings["selected_parameter_manifest_sha256"]
        ),
        "dense_reward_sha256": bindings["dense_reward_sha256"],
        "selected": selected,
        "unselected": unselected,
    }
    recorded_sha = _sha256(identity.get("sha256"), f"{label} SHA-256")
    _require(
        recorded_sha == canonical_sha256(payload),
        f"{label} does not match its partition payload",
    )
    return {
        "schema": "eggroll-es-partitioned-weight-state-v4",
        "sha256": recorded_sha,
        **{key: payload[key] for key in (
            "layer_plan_file_sha256", "layer_plan_sha256",
            "checkpoint_to_runtime_mapping_sha256", "source_unit_count",
            "runtime_selected_name_sha256",
            "selected_parameter_manifest_sha256", "dense_reward_sha256",
            "selected", "unselected",
        )},
    }


def _validate_v4_identity_audit(audit, bindings):
    _require(
        audit.get("schema") == "eggroll-es-alpha-zero-identity-audit-v2",
        "v4 coefficient identity audit schema changed",
    )
    pre_probe = audit.get("pre_probe")
    post_probe = audit.get("post_probe")
    _require(
        isinstance(pre_probe, dict) and pre_probe == post_probe,
        "v4 pre/post alpha-zero identity probes differ",
    )
    _require(
        pre_probe.get("schema") == "eggroll-es-train-only-identity-probe-v4"
        and pre_probe.get("dispatch")
        == "strided_engine_shards_separate_calls"
        and pre_probe.get("reward_config_sha256")
        == bindings["dense_reward_sha256"]
        and pre_probe.get("layer_plan_sha256")
        == bindings["layer_plan_sha256"],
        "v4 identity probe policy or binding changed",
    )
    for key in ("dense_gold_output_sha256", "anchor_output_sha256"):
        _sha256(pre_probe.get(key), f"v4 identity probe {key}")
    _integer(pre_probe.get("domain_requests"), "v4 identity domain requests", minimum=1)
    _integer(pre_probe.get("anchor_requests"), "v4 identity anchor requests", minimum=1)

    reference_states = audit.get("reference_states")
    _require(
        isinstance(reference_states, list) and len(reference_states) == V3_ENGINE_COUNT,
        "v4 identity audit must contain four reference states",
    )
    reference_identities = []
    reference_generations = []
    for rank, nested in enumerate(reference_states):
        reports = _nested_schema_reports(
            nested, "eggroll-es-selected-exact-reference-state-v4",
        )
        _require(len(reports) == 1, f"v4 engine {rank} reference state is ambiguous")
        report = reports[0]
        _require(
            report.get("fresh_for_population") is True,
            f"v4 engine {rank} reference was not fresh",
        )
        for key, value in bindings.items():
            _require(report.get(key) == value, f"v4 engine {rank} reference binding changed")
        reference_generations.append(_integer(
            report.get("reference_generation"),
            f"v4 engine {rank} reference generation", minimum=1,
        ))
        reference_identities.append(_validate_v4_weight_identity(
            report.get("identity"), bindings,
            f"v4 engine {rank} reference identity",
        ))
    _require(
        len(set(reference_generations)) == 1
        and len({canonical_sha256(value) for value in reference_identities}) == 1,
        "v4 engine reference generations or identities differ",
    )

    checks = audit.get("post_reference_checks")
    _require(
        isinstance(checks, list) and len(checks) == V3_ENGINE_COUNT,
        "v4 identity audit must contain four post-reference checks",
    )
    selected_reference = reference_identities[0]["selected"]
    for rank, nested in enumerate(checks):
        reports = _nested_schema_reports(
            nested, "eggroll-es-selected-exact-reference-check-v4",
        )
        _require(len(reports) == 1, f"v4 engine {rank} reference check is ambiguous")
        report = reports[0]
        _require(
            report.get("passed") is True
            and report.get("unselected_audit")
            == "deferred_to_population_completion_v4"
            and _integer(
                report.get("reference_generation"),
                f"v4 engine {rank} checked reference generation", minimum=1,
            ) == reference_generations[0],
            f"v4 engine {rank} post-reference check changed",
        )
        for key, value in bindings.items():
            _require(report.get(key) == value, f"v4 engine {rank} check binding changed")
        reference = _validate_v4_partition_identity(
            report.get("reference"), "selected", bindings,
            f"v4 engine {rank} checked reference",
        )
        current = _validate_v4_partition_identity(
            report.get("current"), "selected", bindings,
            f"v4 engine {rank} checked current",
        )
        _require(
            reference == current == selected_reference,
            f"v4 engine {rank} selected reference check differs",
        )
    return {
        "iteration": _integer(
            audit.get("iteration"), "v4 identity audit iteration", minimum=0,
        ),
        "reference_generation": reference_generations[0],
        "reference_identity": reference_identities[0],
    }


def _validate_identity_audit(audit, *, v4_bindings=None):
    _require(isinstance(audit, dict), "coefficient identity audit is missing")
    _require(
        audit.get("schema") in {
            "eggroll-es-alpha-zero-identity-audit-v2",
            "eggroll-es-alpha-zero-identity-audit-v3",
        },
        "coefficient identity audit schema changed",
    )
    _require(
        audit.get("status") == "passed" and audit.get("passed") is True,
        "coefficient identity audit did not pass",
    )
    _require(
        audit.get("training_signal")
        == "train_batch_and_train_only_anchor_only",
        "identity audit training signal changed",
    )
    if v4_bindings is not None:
        return _validate_v4_identity_audit(audit, v4_bindings)
    pre_probe = audit.get("pre_probe")
    post_probe = audit.get("post_probe")
    _require(
        isinstance(pre_probe, dict) and pre_probe == post_probe,
        "pre/post alpha-zero identity probes differ",
    )
    for key in ("domain_output_sha256", "anchor_output_sha256"):
        _sha256(pre_probe.get(key), f"identity probe {key}")
    _integer(pre_probe.get("domain_requests"), "identity domain requests", minimum=1)
    _integer(pre_probe.get("anchor_requests"), "identity anchor requests", minimum=1)

    reference_states = audit.get("reference_states")
    _require(
        isinstance(reference_states, list) and len(reference_states) == 4,
        "identity audit must contain four engine reference states",
    )
    reference_hashes = []
    for engine_index, state in enumerate(reference_states):
        leaves = list(_leaf_dicts(state))
        hashes = [leaf.get("sha256") for leaf in leaves if "sha256" in leaf]
        _require(hashes, f"engine {engine_index} reference hash is missing")
        for value in hashes:
            reference_hashes.append(_sha256(value, "reference weight SHA-256"))
    _require(len(set(reference_hashes)) == 1, "engine reference weight hashes differ")

    checks = audit.get("post_reference_checks")
    _require(
        isinstance(checks, list) and len(checks) == 4,
        "identity audit must contain four post-reference checks",
    )
    for engine_index, check in enumerate(checks):
        leaves = [leaf for leaf in _leaf_dicts(check) if "passed" in leaf]
        _require(leaves, f"engine {engine_index} post-reference check is missing")
        _require(
            all(leaf.get("passed") is True for leaf in leaves),
            f"engine {engine_index} post-reference check failed",
        )
        for leaf in leaves:
            for side in ("reference", "current"):
                _require(isinstance(leaf.get(side), dict), "weight check identity is missing")
                _sha256(leaf[side].get("sha256"), f"weight check {side} SHA-256")
    return None


def _validate_v3_preflight(preflight, label):
    _require(isinstance(preflight, dict), f"{label} is missing")
    _require(
        preflight.get("schema") == "eggroll-es-local-allocation-preflight-v3",
        f"{label} schema changed",
    )
    _require(preflight.get("passed") is True, f"{label} did not pass")
    _require(
        preflight.get("collectives_created") is False
        and preflight.get("rng_consumed") is False
        and preflight.get("weights_changed") is False,
        f"{label} was not local and side-effect free",
    )
    _require(
        preflight.get("scratch_freed_before_collectives") is True,
        f"{label} retained scratch before collectives",
    )
    _require(
        preflight.get("accumulator_dtype") == "torch.float32",
        f"{label} accumulator was not FP32",
    )
    _integer(
        preflight.get("simulated_peak_temporary_bytes"),
        f"{label} simulated temporary bytes", minimum=1,
    )
    _integer(
        preflight.get("parameter_count"),
        f"{label} parameter count", minimum=1,
    )
    _require(
        isinstance(preflight.get("largest_parameter_name"), str)
        and bool(preflight["largest_parameter_name"]),
        f"{label} largest parameter name is missing",
    )
    _require(
        isinstance(preflight.get("largest_parameter_shape"), list),
        f"{label} largest parameter shape is missing",
    )
    _require(
        isinstance(preflight.get("parameter_dtype"), str)
        and bool(preflight["parameter_dtype"]),
        f"{label} parameter dtype is missing",
    )


def _v3_update_manifest(
    *, coefficient_sha256, population_size, reference_generation, plan_id,
    update_sequence, previous_alpha, target_alpha, expected_base_sha256,
):
    return {
        "schema": "eggroll-es-distributed-update-manifest-v3",
        "coefficient_sha256": coefficient_sha256,
        "population_size": population_size,
        "world_size": V3_ENGINE_COUNT,
        "reference_generation": reference_generation,
        "plan_id": plan_id,
        "update_sequence": update_sequence,
        "previous_alpha": previous_alpha,
        "target_alpha": target_alpha,
        "expected_base_sha256": expected_base_sha256,
    }


def _validate_v3_rank_set(reports, label, *, communicator_rank=False):
    _require(
        isinstance(reports, list) and len(reports) == V3_ENGINE_COUNT,
        f"{label} must contain four engine reports",
    )
    _require(
        all(isinstance(report, dict) for report in reports),
        f"{label} contains a non-object report",
    )
    ranks = [
        (
            report.get("communicator", {}).get("rank")
            if communicator_rank and isinstance(report.get("communicator"), dict)
            else report.get("rank")
        )
        for report in reports
    ]
    _require(
        all(not isinstance(rank, bool) and isinstance(rank, int) for rank in ranks)
        and sorted(ranks) == list(range(V3_ENGINE_COUNT)),
        f"{label} ranks are not exactly 0,1,2,3",
    )
    return {rank: report for rank, report in zip(ranks, reports)}


def _validate_v3_distributed_provenance(
    coefficient_plan, *, coefficient_sha, targets, snapshot, journal_seeds,
):
    population_size = snapshot["recipe"]["population_size"]
    _require(
        isinstance(journal_seeds, list)
        and len(journal_seeds) == population_size,
        "v3 journal population seeds are incomplete",
    )
    for seed in journal_seeds:
        _integer(seed, "v3 population seed", minimum=0)
    _require(
        len(set(journal_seeds)) == population_size,
        "v3 population seeds are not unique",
    )
    raw_coefficients = coefficient_plan.get("coefficients")
    _require(
        isinstance(raw_coefficients, list)
        and len(raw_coefficients) == population_size,
        "v3 canonical coefficient values are incomplete",
    )
    coefficients = [
        _finite_float(value, "v3 canonical coefficient")
        for value in raw_coefficients
    ]
    _require(
        canonical_sha256({
            "seeds": journal_seeds,
            "coefficients": coefficients,
        }) == coefficient_sha,
        "v3 canonical coefficient values do not match their SHA-256",
    )

    metadata = coefficient_plan.get("distributed_update_v3")
    _require(isinstance(metadata, dict), "v3 distributed seed plan is missing")
    _require(
        metadata.get("schema") == "eggroll-es-distributed-seed-plan-v3",
        "v3 distributed seed plan schema changed",
    )
    _require_exact_fields(
        metadata,
        {
            "engine_count": V3_ENGINE_COUNT,
            "tp_per_engine": 1,
            "seed_sharding": "strided_by_inter_engine_rank",
            "collective_dtype": "torch.float32",
            "reference_recapture_policy": "once_before_next_population_only",
        },
        "v3 distributed seed plan",
    )
    plan_id = _sha256(metadata.get("plan_id"), "v3 plan ID")
    reference_generation = _integer(
        metadata.get("reference_generation"),
        "v3 reference generation", minimum=1,
    )
    reference_identity = _validate_weight_identity(
        metadata.get("reference_identity"), "v3 reference identity",
    )
    audit = coefficient_plan["identity_audit"]
    expected_plan_id = canonical_sha256({
        "schema": "eggroll-es-distributed-plan-id-v3",
        "iteration": _integer(
            audit.get("iteration"), "v3 identity audit iteration", minimum=0,
        ),
        "coefficient_sha256": coefficient_sha,
        "reference_generation": reference_generation,
        "reference_sha256": reference_identity["sha256"],
    })
    _require(plan_id == expected_plan_id, "v3 plan ID does not match its inputs")

    applications = coefficient_plan.get("applications")
    expected_count = sum(target > 0.0 for target in targets)
    _require(
        isinstance(applications, list) and len(applications) == expected_count,
        "v3 application count does not match nonzero targets",
    )
    previous_alpha = 0.0
    expected_base_sha = reference_identity["sha256"]
    for sequence, (target_alpha, application) in enumerate(
        zip(targets[1:], applications), 1,
    ):
        label = f"v3 application {sequence}"
        _require(isinstance(application, dict), f"{label} is invalid")
        _require(
            application.get("schema")
            == "eggroll-es-distributed-alpha-application-v3",
            f"{label} schema changed",
        )
        _require(
            _integer(
                application.get("update_sequence"),
                f"{label} sequence", minimum=1,
            ) == sequence,
            f"{label} sequence changed",
        )
        recorded_target_alpha = _finite_float(
            application.get("target_alpha"), f"{label} target alpha",
        )
        recorded_increment = _finite_float(
            application.get("alpha_increment"), f"{label} alpha increment",
        )
        _require(
            _same_float(recorded_target_alpha, target_alpha)
            and _same_float(recorded_increment, target_alpha - previous_alpha),
            f"{label} alpha transition changed",
        )
        _require(
            application.get("coefficient_sha256") == coefficient_sha,
            f"{label} coefficient identity changed",
        )
        manifest_sha = _sha256(
            application.get("manifest_sha256"), f"{label} manifest SHA-256",
        )
        expected_manifest = _v3_update_manifest(
            coefficient_sha256=coefficient_sha,
            population_size=population_size,
            reference_generation=reference_generation,
            plan_id=plan_id,
            update_sequence=sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=expected_base_sha,
        )
        _require(
            manifest_sha == canonical_sha256(expected_manifest),
            f"{label} manifest does not match the committed transition",
        )

        prepared = _validate_v3_rank_set(
            application.get("prepared_shards"), f"{label} prepared reports",
        )
        for rank, report in prepared.items():
            _require(
                report.get("schema")
                == "eggroll-es-distributed-update-prepared-v3"
                and report.get("prepared") is True,
                f"{label} rank {rank} was not prepared",
            )
            prepared_generation = _integer(
                report.get("reference_generation"),
                f"{label} rank {rank} prepared reference generation",
                minimum=1,
            )
            prepared_sequence = _integer(
                report.get("update_sequence"),
                f"{label} rank {rank} prepared sequence", minimum=1,
            )
            _require(
                report.get("manifest_sha256") == manifest_sha
                and report.get("world_size") == V3_ENGINE_COUNT
                and prepared_generation == reference_generation
                and prepared_sequence == sequence
                and report.get("base_sha256") == expected_base_sha,
                f"{label} rank {rank} prepared metadata differs",
            )
            expected_indices = list(range(rank, population_size, V3_ENGINE_COUNT))
            _require(
                isinstance(report.get("shard_indices"), list)
                and all(
                    not isinstance(index, bool) and isinstance(index, int)
                    for index in report["shard_indices"]
                )
                and report.get("shard_indices") == expected_indices
                and report.get("shard_seeds")
                == [journal_seeds[index] for index in expected_indices],
                f"{label} rank {rank} seed shard changed",
            )
            expected_shard_pair_sha = canonical_sha256({
                "seeds": [journal_seeds[index] for index in expected_indices],
                "coefficients": [
                    coefficients[index] for index in expected_indices
                ],
            })
            _require(
                report.get("shard_pair_sha256") == expected_shard_pair_sha,
                f"{label} rank {rank} seed/coefficient shard SHA-256 differs",
            )
            _validate_v3_preflight(
                report.get("allocation_preflight"),
                f"{label} rank {rank} allocation preflight",
            )

        final_identity = _validate_weight_identity(
            application.get("final_identity"), f"{label} final identity",
        )
        executed = _validate_v3_rank_set(
            application.get("executed_collectives"),
            f"{label} executed reports",
        )
        executed_parameter_counts = set()
        executed_element_counts = set()
        for rank, report in executed.items():
            _require(
                report.get("schema")
                == "eggroll-es-distributed-update-executed-v3"
                and report.get("executed") is True,
                f"{label} rank {rank} did not execute",
            )
            _require(
                report.get("manifest_sha256") == manifest_sha
                and report.get("world_size") == V3_ENGINE_COUNT
                and report.get("collective_dtype") == "torch.float32",
                f"{label} rank {rank} collective metadata differs",
            )
            parameter_count = _integer(
                report.get("parameter_count"),
                f"{label} rank {rank} parameter count", minimum=1,
            )
            element_count = _integer(
                report.get("reduced_element_count"),
                f"{label} rank {rank} reduced elements", minimum=1,
            )
            executed_parameter_counts.add(parameter_count)
            executed_element_counts.add(element_count)
            _require(
                _validate_weight_identity(
                    report.get("final_identity"),
                    f"{label} rank {rank} final identity",
                ) == final_identity,
                f"{label} rank {rank} final identity differs",
            )
        _require(
            executed_parameter_counts == {final_identity["parameter_count"]},
            f"{label} executed parameter counts differ",
        )
        _require(
            len(executed_element_counts) == 1,
            f"{label} executed element counts differ",
        )

        commits = _validate_v3_rank_set(
            application.get("commits"), f"{label} commit reports",
        )
        for rank, report in commits.items():
            _require(
                report.get("schema")
                == "eggroll-es-distributed-update-committed-v3"
                and report.get("committed") is True,
                f"{label} rank {rank} did not commit",
            )
            commit_generation = _integer(
                report.get("reference_generation"),
                f"{label} rank {rank} commit reference generation",
                minimum=1,
            )
            commit_sequence = _integer(
                report.get("update_sequence"),
                f"{label} rank {rank} commit sequence", minimum=1,
            )
            commit_alpha = _finite_float(
                report.get("accepted_alpha"),
                f"{label} rank {rank} committed alpha",
            )
            _require(
                report.get("manifest_sha256") == manifest_sha
                and report.get("final_sha256") == final_identity["sha256"]
                and commit_generation == reference_generation
                and report.get("reference_fresh_for_population") is False
                and commit_sequence == sequence
                and _same_float(commit_alpha, target_alpha),
                f"{label} rank {rank} commit metadata differs",
            )

        post_states = _validate_v3_rank_set(
            application.get("post_commit_states"),
            f"{label} post-commit states",
            communicator_rank=True,
        )
        for rank, state in post_states.items():
            communicator = state.get("communicator")
            _require(
                state.get("schema") == "eggroll-es-distributed-worker-state-v3"
                and state.get("pending") is False
                and isinstance(communicator, dict)
                and communicator.get("rank") == rank
                and communicator.get("world_size") == V3_ENGINE_COUNT
                and communicator.get("tp_world_size") == 1
                and communicator.get("available") is True
                and communicator.get("disabled") is False,
                f"{label} rank {rank} post-commit communicator differs",
            )
            state_generation = _integer(
                state.get("reference_generation"),
                f"{label} rank {rank} state reference generation", minimum=1,
            )
            state_sequence = _integer(
                state.get("update_sequence"),
                f"{label} rank {rank} state sequence", minimum=1,
            )
            state_alpha = _finite_float(
                state.get("accepted_alpha"),
                f"{label} rank {rank} state accepted alpha",
            )
            _require(
                state_generation == reference_generation
                and state.get("reference_fresh_for_population") is False
                and state.get("update_session") == plan_id
                and state_sequence == sequence
                and _same_float(state_alpha, target_alpha),
                f"{label} rank {rank} post-commit state differs",
            )
            _require(
                _validate_weight_identity(
                    state.get("reference_identity"),
                    f"{label} rank {rank} retained reference",
                ) == reference_identity,
                f"{label} rank {rank} retained reference changed",
            )
            _require(
                _validate_weight_identity(
                    state.get("current_identity"),
                    f"{label} rank {rank} current identity",
                ) == final_identity,
                f"{label} rank {rank} current identity differs",
            )

        _require(
            application.get("reference_recaptured") is False
            and application.get("reference_fresh_for_population") is False,
            f"{label} improperly recaptured the population reference",
        )
        _require(
            application.get("bf16_alpha_semantics")
            == "path_dependent_monotonic_increment"
            and application.get("direct_alpha_confirmation_required") is True,
            f"{label} direct-confirmation policy changed",
        )
        previous_alpha = target_alpha
        expected_base_sha = final_identity["sha256"]

    return {
        "plan_id": plan_id,
        "reference_generation": reference_generation,
        "application_count": len(applications),
        "final_identity_sha256": expected_base_sha,
    }


def _v4_update_manifest(
    *, coefficient_sha256, population_size, reference_generation, plan_id,
    update_sequence, previous_alpha, target_alpha, expected_base_sha256,
    bindings,
):
    return {
        "schema": "eggroll-es-layer-restricted-update-manifest-v4",
        "coefficient_sha256": coefficient_sha256,
        "population_size": population_size,
        "world_size": V3_ENGINE_COUNT,
        "reference_generation": reference_generation,
        "plan_id": plan_id,
        "update_sequence": update_sequence,
        "previous_alpha": previous_alpha,
        "target_alpha": target_alpha,
        "expected_base_sha256": expected_base_sha256,
        **bindings,
    }


def _validate_v4_boundary_audit(
    audit, *, bindings, audit_iteration, reference_generation,
    reference_identity,
):
    _require(isinstance(audit, dict), "v4 population boundary audit is missing")
    _require(
        audit.get("schema") == "eggroll-es-population-boundary-audit-v4"
        and audit.get("phase")
        == "after_complete_population_exact_restore_before_plan"
        and audit.get("passed") is True,
        "v4 population boundary audit policy changed or did not pass",
    )
    _require(
        _integer(audit.get("iteration"), "v4 boundary iteration", minimum=0)
        == audit_iteration
        and audit.get("engine_count") == V3_ENGINE_COUNT
        and type(audit.get("engine_count")) is int
        and _integer(
            audit.get("reference_generation"),
            "v4 boundary reference generation", minimum=1,
        ) == reference_generation,
        "v4 population boundary audit identity changed",
    )
    _require(
        audit.get("runtime_mapping") == bindings
        and audit.get("unselected_origin_sha256")
        == bindings["unselected_origin_sha256"],
        "v4 population boundary runtime binding changed",
    )
    recorded_reference = _validate_v4_weight_identity(
        audit.get("reference_identity"), bindings,
        "v4 boundary reference identity",
    )
    recorded_current = _validate_v4_weight_identity(
        audit.get("current_identity"), bindings,
        "v4 boundary current identity",
    )
    _require(
        recorded_reference == recorded_current == reference_identity,
        "v4 population boundary did not exactly restore the reference",
    )
    reports = _validate_v3_rank_set(
        audit.get("worker_reports"), "v4 population boundary worker reports",
    )
    for rank, report in reports.items():
        _require(
            report.get("schema") == "eggroll-es-post-population-audit-v4"
            and report.get("passed") is True
            and report.get("world_size") == V3_ENGINE_COUNT
            and type(report.get("world_size")) is int
            and _integer(
                report.get("reference_generation"),
                f"v4 population boundary rank {rank} reference generation",
                minimum=1,
            ) == reference_generation
            and report.get("reference_sha256") == reference_identity["sha256"],
            f"v4 population boundary rank {rank} metadata changed",
        )
        for key, value in bindings.items():
            _require(
                report.get(key) == value,
                f"v4 population boundary rank {rank} binding changed",
            )
        _require(
            _validate_v4_weight_identity(
                report.get("current_identity"), bindings,
                f"v4 population boundary rank {rank} current identity",
            ) == reference_identity,
            f"v4 population boundary rank {rank} did not restore reference",
        )
    audit_sha = _sha256(audit.get("audit_sha256"), "v4 boundary audit SHA-256")
    unhashed = copy.deepcopy(audit)
    unhashed.pop("audit_sha256", None)
    _require(
        audit_sha == canonical_sha256(unhashed),
        "v4 population boundary audit hash does not match",
    )
    return audit_sha


def _validate_v4_preflight(preflight, bindings, label):
    _require(isinstance(preflight, dict), f"{label} is missing")
    _require(
        preflight.get("schema") == "eggroll-es-selected-allocation-preflight-v4"
        and preflight.get("passed") is True
        and preflight.get("parameter_count") == V4_RUNTIME_PARAMETER_COUNT
        and type(preflight.get("parameter_count")) is int
        and preflight.get("element_count") == V4_SELECTED_ELEMENT_COUNT
        and type(preflight.get("element_count")) is int
        and preflight.get("accumulator_dtype") == "torch.float32"
        and preflight.get("scratch_freed_before_collectives") is True
        and preflight.get("collectives_created") is False
        and preflight.get("rng_consumed") is False
        and preflight.get("weights_changed") is False,
        f"{label} was not selected-only and side-effect free",
    )
    _integer(
        preflight.get("simulated_peak_temporary_bytes"),
        f"{label} simulated temporary bytes", minimum=1,
    )
    _require(
        isinstance(preflight.get("largest_parameter_name"), str)
        and bool(preflight["largest_parameter_name"])
        and isinstance(preflight.get("largest_parameter_shape"), list)
        and preflight["largest_parameter_shape"]
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in preflight["largest_parameter_shape"]
        )
        and preflight.get("parameter_dtype") == "torch.bfloat16",
        f"{label} selected parameter metadata changed",
    )
    for key, value in bindings.items():
        _require(preflight.get(key) == value, f"{label} binding changed")


def _validate_v4_distributed_provenance(
    coefficient_plan, *, coefficient_sha, targets, snapshot, journal_seeds,
):
    population_size = snapshot["recipe"]["population_size"]
    _require(
        isinstance(journal_seeds, list)
        and len(journal_seeds) == population_size,
        "v4 journal population seeds are incomplete",
    )
    for seed in journal_seeds:
        _integer(seed, "v4 population seed", minimum=0)
    _require(
        len(set(journal_seeds)) == population_size,
        "v4 population seeds are not unique",
    )
    raw_coefficients = coefficient_plan.get("coefficients")
    _require(
        isinstance(raw_coefficients, list)
        and len(raw_coefficients) == population_size,
        "v4 canonical coefficient values are incomplete",
    )
    coefficients = [
        _finite_float(value, "v4 canonical coefficient")
        for value in raw_coefficients
    ]
    _require(
        canonical_sha256({
            "seeds": journal_seeds,
            "coefficients": coefficients,
        }) == coefficient_sha,
        "v4 canonical coefficient values do not match their SHA-256",
    )

    layer = _validate_v4_layer_identity(
        coefficient_plan.get("frozen_layer_plan_v4"),
        "v4 coefficient frozen layer plan", require_runtime=True,
    )
    for key in (
        "path", "file_sha256", "plan_sha256", "model_config_path",
        "model_config_sha256",
    ):
        _require(
            layer[key] == snapshot["frozen_layer_plan_v4"][key],
            f"v4 coefficient layer plan {key} differs from snapshot",
        )
    bindings = layer["runtime_mapping"]
    reward = _validate_v4_dense_reward(
        coefficient_plan.get("dense_gold_reward_v4"),
        "v4 coefficient dense reward",
    )
    _require(
        reward == snapshot["dense_gold_reward_v4"],
        "v4 coefficient dense reward differs from snapshot",
    )
    identity_audit = _validate_identity_audit(
        coefficient_plan.get("identity_audit"), v4_bindings=bindings,
    )

    metadata = coefficient_plan.get("distributed_update_v4")
    _require(isinstance(metadata, dict), "v4 distributed seed plan is missing")
    _require(
        metadata.get("schema") == "eggroll-es-distributed-seed-plan-v4",
        "v4 distributed seed plan schema changed",
    )
    _require_exact_fields(
        metadata,
        {
            "engine_count": V3_ENGINE_COUNT,
            "tp_per_engine": 1,
            "seed_sharding": "strided_by_inter_engine_rank",
            "collective_dtype": "torch.float32",
            "reference_recapture_policy": "once_before_next_population_only",
            "layer_plan_sha256": bindings["layer_plan_sha256"],
            "dense_reward_sha256": bindings["dense_reward_sha256"],
            "runtime_mapping": bindings,
            "runtime_mapping_sha256": canonical_sha256(bindings),
        },
        "v4 distributed seed plan",
    )
    plan_id = _sha256(metadata.get("plan_id"), "v4 plan ID")
    reference_generation = _integer(
        metadata.get("reference_generation"),
        "v4 reference generation", minimum=1,
    )
    reference_identity = _validate_v4_weight_identity(
        metadata.get("reference_identity"), bindings,
        "v4 reference identity",
    )
    _require(
        reference_generation == identity_audit["reference_generation"]
        and reference_identity == identity_audit["reference_identity"],
        "v4 distributed reference differs from identity audit",
    )
    boundary_sha = _validate_v4_boundary_audit(
        coefficient_plan.get("population_boundary_audit_v4"),
        bindings=bindings,
        audit_iteration=identity_audit["iteration"],
        reference_generation=reference_generation,
        reference_identity=reference_identity,
    )
    _require(
        metadata.get("population_boundary_audit_sha256") == boundary_sha,
        "v4 distributed plan boundary-audit identity changed",
    )
    expected_plan_id = canonical_sha256({
        "schema": "eggroll-es-distributed-plan-id-v4",
        "iteration": identity_audit["iteration"],
        "coefficient_sha256": coefficient_sha,
        "reference_generation": reference_generation,
        "reference_sha256": reference_identity["sha256"],
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "reward_config_sha256": bindings["dense_reward_sha256"],
        "runtime_mapping_sha256": canonical_sha256(bindings),
        "population_boundary_audit_sha256": boundary_sha,
    })
    _require(plan_id == expected_plan_id, "v4 plan ID does not match its inputs")

    applications = coefficient_plan.get("applications")
    expected_count = sum(target > 0.0 for target in targets)
    _require(
        isinstance(applications, list) and len(applications) == expected_count,
        "v4 application count does not match nonzero targets",
    )
    previous_alpha = 0.0
    expected_base_sha = reference_identity["sha256"]
    for sequence, (target_alpha, application) in enumerate(
        zip(targets[1:], applications), 1,
    ):
        label = f"v4 application {sequence}"
        _require(isinstance(application, dict), f"{label} is invalid")
        _require(
            application.get("schema")
            == "eggroll-es-distributed-alpha-application-v4"
            and application.get("update_sequence") == sequence
            and type(application.get("update_sequence")) is int
            and _same_float(application.get("target_alpha"), target_alpha)
            and _same_float(
                application.get("alpha_increment"),
                target_alpha - previous_alpha,
            )
            and application.get("coefficient_sha256") == coefficient_sha
            and application.get("layer_plan_sha256")
            == bindings["layer_plan_sha256"]
            and application.get("dense_reward_sha256")
            == bindings["dense_reward_sha256"]
            and application.get("runtime_mapping") == bindings,
            f"{label} plan, reward, or alpha binding changed",
        )
        expected_manifest = _v4_update_manifest(
            coefficient_sha256=coefficient_sha,
            population_size=population_size,
            reference_generation=reference_generation,
            plan_id=plan_id,
            update_sequence=sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=expected_base_sha,
            bindings=bindings,
        )
        _require(
            application.get("manifest") == expected_manifest,
            f"{label} manifest payload changed",
        )
        manifest_sha = _sha256(
            application.get("manifest_sha256"), f"{label} manifest SHA-256",
        )
        _require(
            manifest_sha == canonical_sha256(expected_manifest)
            and manifest_sha == canonical_sha256(application["manifest"]),
            f"{label} manifest does not match the committed transition",
        )

        prepared = _validate_v3_rank_set(
            application.get("prepared_shards"), f"{label} prepared reports",
        )
        for rank, report in prepared.items():
            _require(
                report.get("schema")
                == "eggroll-es-layer-restricted-update-prepared-v4"
                and report.get("prepared") is True
                and report.get("manifest_sha256") == manifest_sha
                and report.get("world_size") == V3_ENGINE_COUNT
                and type(report.get("world_size")) is int
                and _integer(
                    report.get("reference_generation"),
                    f"{label} rank {rank} prepared reference generation",
                    minimum=1,
                ) == reference_generation
                and _integer(
                    report.get("update_sequence"),
                    f"{label} rank {rank} prepared sequence", minimum=1,
                ) == sequence
                and report.get("base_sha256") == expected_base_sha,
                f"{label} rank {rank} prepared metadata changed",
            )
            for key, value in bindings.items():
                _require(report.get(key) == value, f"{label} rank {rank} prepared binding changed")
            expected_indices = list(range(rank, population_size, V3_ENGINE_COUNT))
            _require(
                report.get("shard_indices") == expected_indices
                and report.get("shard_seeds")
                == [journal_seeds[index] for index in expected_indices]
                and report.get("shard_pair_sha256") == canonical_sha256({
                    "seeds": [journal_seeds[index] for index in expected_indices],
                    "coefficients": [
                        coefficients[index] for index in expected_indices
                    ],
                }),
                f"{label} rank {rank} seed/coefficient shard changed",
            )
            _validate_v4_preflight(
                report.get("allocation_preflight"), bindings,
                f"{label} rank {rank} allocation preflight",
            )

        final_identity = _validate_v4_weight_identity(
            application.get("final_identity"), bindings,
            f"{label} final identity",
        )
        _require(
            final_identity["unselected"] == reference_identity["unselected"],
            f"{label} unselected partition metadata changed",
        )
        executed = _validate_v3_rank_set(
            application.get("executed_collectives"),
            f"{label} executed reports",
        )
        for rank, report in executed.items():
            _require(
                report.get("schema")
                == "eggroll-es-layer-restricted-update-executed-v4"
                and report.get("executed") is True
                and report.get("manifest_sha256") == manifest_sha
                and report.get("world_size") == V3_ENGINE_COUNT
                and type(report.get("world_size")) is int
                and report.get("collective_dtype") == "torch.float32"
                and report.get("parameter_count") == V4_RUNTIME_PARAMETER_COUNT
                and type(report.get("parameter_count")) is int
                and report.get("reduced_element_count")
                == V4_SELECTED_ELEMENT_COUNT
                and type(report.get("reduced_element_count")) is int,
                f"{label} rank {rank} collective metadata changed",
            )
            for key, value in bindings.items():
                _require(report.get(key) == value, f"{label} rank {rank} executed binding changed")
            _require(
                _validate_v4_weight_identity(
                    report.get("final_identity"), bindings,
                    f"{label} rank {rank} final identity",
                ) == final_identity,
                f"{label} rank {rank} final identity differs",
            )

        commits = _validate_v3_rank_set(
            application.get("commits"), f"{label} commit reports",
        )
        for rank, report in commits.items():
            _require(
                report.get("schema")
                == "eggroll-es-layer-restricted-update-committed-v4"
                and report.get("committed") is True
                and report.get("manifest_sha256") == manifest_sha
                and report.get("final_sha256") == final_identity["sha256"]
                and _integer(
                    report.get("reference_generation"),
                    f"{label} rank {rank} commit reference generation",
                    minimum=1,
                ) == reference_generation
                and report.get("reference_fresh_for_population") is False
                and _integer(
                    report.get("update_sequence"),
                    f"{label} rank {rank} commit sequence", minimum=1,
                ) == sequence
                and _same_float(report.get("accepted_alpha"), target_alpha),
                f"{label} rank {rank} commit metadata changed",
            )
            for key, value in bindings.items():
                _require(report.get(key) == value, f"{label} rank {rank} commit binding changed")

        post_states = _validate_v3_rank_set(
            application.get("post_commit_states"),
            f"{label} post-commit states", communicator_rank=True,
        )
        for rank, state in post_states.items():
            communicator = state.get("communicator")
            _require(
                state.get("schema")
                == "eggroll-es-layer-restricted-worker-state-v4"
                and state.get("pending") is False
                and isinstance(communicator, dict)
                and communicator.get("rank") == rank
                and communicator.get("world_size") == V3_ENGINE_COUNT
                and type(communicator.get("world_size")) is int
                and communicator.get("tp_world_size") == 1
                and type(communicator.get("tp_world_size")) is int
                and communicator.get("available") is True
                and communicator.get("disabled") is False
                and _integer(
                    state.get("reference_generation"),
                    f"{label} rank {rank} state reference generation",
                    minimum=1,
                ) == reference_generation
                and state.get("reference_fresh_for_population") is False
                and state.get("update_session") == plan_id
                and _integer(
                    state.get("update_sequence"),
                    f"{label} rank {rank} state sequence", minimum=1,
                ) == sequence
                and _same_float(state.get("accepted_alpha"), target_alpha),
                f"{label} rank {rank} post-commit state changed",
            )
            for key, value in bindings.items():
                _require(state.get(key) == value, f"{label} rank {rank} state binding changed")
            _require(
                _validate_v4_weight_identity(
                    state.get("reference_identity"), bindings,
                    f"{label} rank {rank} retained reference",
                ) == reference_identity,
                f"{label} rank {rank} retained reference changed",
            )
            _require(
                _validate_v4_weight_identity(
                    state.get("current_identity"), bindings,
                    f"{label} rank {rank} current identity",
                ) == final_identity,
                f"{label} rank {rank} current identity differs",
            )

        _require(
            application.get("reference_recaptured") is False
            and application.get("reference_fresh_for_population") is False
            and application.get("bf16_alpha_semantics")
            == "path_dependent_monotonic_increment"
            and application.get("direct_alpha_confirmation_required") is True,
            f"{label} reference or direct-confirmation policy changed",
        )
        previous_alpha = target_alpha
        expected_base_sha = final_identity["sha256"]

    return {
        "plan_id": plan_id,
        "reference_generation": reference_generation,
        "application_count": len(applications),
        "final_identity_sha256": expected_base_sha,
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "runtime_mapping_sha256": canonical_sha256(bindings),
        "dense_reward_sha256": bindings["dense_reward_sha256"],
        "population_boundary_audit_sha256": boundary_sha,
    }


def validate_journal(journal):
    _require(isinstance(journal, dict), "journal root must be an object")
    _assert_no_heldout(journal)
    _verify_content_hash(journal)
    journal_schema = journal.get("schema")
    _require(
        journal_schema in ALLOWED_SCHEMAS,
        "journal is not corrected v2/v3/v4",
    )
    _require(journal.get("status") == "complete", "journal is incomplete or failed")
    _require(journal.get("in_progress") is None, "journal still has in-progress work")
    policy = journal.get("policy")
    _require(isinstance(policy, dict), "line-search policy is missing")
    _require(
        policy.get("alpha_order") == "zero_then_strictly_increasing"
        and policy.get("branching") is False
        and policy.get("resume") is False
        and policy.get("rollback") is False
        and policy.get("selection_during_execution") is False,
        "line-search execution policy changed",
    )
    _require(
        policy.get("ood_qa_max_degradation") == 0.0
        and policy.get("ood_prose_max_degradation") == 0.0,
        "line-search OOD policy is not strict",
    )

    snapshot = _validate_snapshot(journal.get("snapshot"))
    expected_snapshot_schema = {
        "eggroll-es-anchor-alpha-line-search-v2": (
            "eggroll-es-anchor-line-search-snapshot-v2"
        ),
        "eggroll-es-anchor-alpha-line-search-v3": (
            "eggroll-es-anchor-line-search-snapshot-v3"
        ),
        "eggroll-es-anchor-alpha-line-search-v4": (
            "eggroll-es-anchor-line-search-snapshot-v4"
        ),
    }[journal_schema]
    _require(
        snapshot["schema"] == expected_snapshot_schema,
        "journal and snapshot corrected-version schemas differ",
    )
    if journal_schema in {
        "eggroll-es-anchor-alpha-line-search-v3",
        "eggroll-es-anchor-alpha-line-search-v4",
    }:
        _require_exact_fields(
            policy,
            {
                "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
                "direct_alpha_confirmation_required": True,
            },
            "v3 journal policy",
        )
    if journal_schema == "eggroll-es-anchor-alpha-line-search-v4":
        _require_exact_fields(
            policy,
            {
                "frozen_layer_plan_required": True,
                "dense_gold_reward_required": True,
            },
            "v4 journal policy",
        )
    targets = journal.get("targets")
    _require(isinstance(targets, list), "journal targets are missing")
    targets = [_finite_float(value, "journal target alpha") for value in targets]
    _require(targets == snapshot["targets"], "journal and recipe targets differ")
    states = journal.get("states")
    _require(
        isinstance(states, list) and len(states) == len(targets),
        "completed state count does not match targets",
    )

    configuration = journal.get("trainer_configuration")
    _require(isinstance(configuration, dict), "trainer configuration is missing")
    config_pairs = {
        "model_name": "model_name", "sigma": "sigma",
        "population_size": "population_size", "batch_size": "batch_size",
        "mini_batch_size": "mini_batch_size", "max_tokens": "max_tokens",
        "global_seed": "seed", "min_anchor_cosine": "min_anchor_cosine",
        "anchor_items_per_step": "anchor_items_per_step",
    }
    for config_key, recipe_key in config_pairs.items():
        _require(
            configuration.get(config_key) == snapshot["recipe"].get(recipe_key),
            f"trainer configuration {config_key} differs from recipe",
        )

    coefficient_plan = journal.get("coefficient_plan")
    _require(isinstance(coefficient_plan, dict), "coefficient plan is missing")
    coefficient_sha = _sha256(
        coefficient_plan.get("coefficient_sha256"),
        "coefficient plan SHA-256",
    )
    _require(
        coefficient_plan.get("seed_count") == snapshot["recipe"]["population_size"],
        "coefficient seed count differs from population size",
    )
    if journal_schema != "eggroll-es-anchor-alpha-line-search-v4":
        _validate_identity_audit(coefficient_plan.get("identity_audit"))
    distributed_v3 = None
    distributed_v4 = None
    if journal_schema == "eggroll-es-anchor-alpha-line-search-v3":
        distributed_v3 = _validate_v3_distributed_provenance(
            coefficient_plan,
            coefficient_sha=coefficient_sha,
            targets=targets,
            snapshot=snapshot,
            journal_seeds=journal.get("seeds"),
        )
    elif journal_schema == "eggroll-es-anchor-alpha-line-search-v4":
        distributed_v4 = _validate_v4_distributed_provenance(
            coefficient_plan,
            coefficient_sha=coefficient_sha,
            targets=targets,
            snapshot=snapshot,
            journal_seeds=journal.get("seeds"),
        )

    baseline = None
    normalized_states = []
    for index, (target, state) in enumerate(zip(targets, states)):
        _require(isinstance(state, dict), f"state {index} is invalid")
        _require(state.get("state_index") == index, f"state {index} index changed")
        _require(state.get("eval_iteration") == index, f"state {index} eval iteration changed")
        _require(
            _same_float(state.get("target_alpha"), target),
            f"state {index} target alpha changed",
        )
        expected_increment = target if index == 0 else target - targets[index - 1]
        _require(
            _same_float(state.get("alpha_increment"), expected_increment),
            f"state {index} alpha increment is not the monotonic path increment",
        )
        _require(
            state.get("coefficient_sha256") == coefficient_sha,
            f"state {index} coefficient identity changed",
        )
        qa = state.get("qa")
        _require(isinstance(qa, dict), f"state {index} QA results are missing")
        _require(
            len(qa) == len(ALLOWED_EVAL_SPLITS)
            and set(qa) == set(ALLOWED_EVAL_SPLITS),
            f"state {index} QA splits are not allowlisted",
        )
        normalized_qa = {
            split: _validate_qa_summary(qa[split], f"state {index} {split}")
            for split in ALLOWED_EVAL_SPLITS
        }
        prose = _validate_prose_summary(state.get("ood_prose"), f"state {index}")
        if index == 0:
            baseline = {"qa": normalized_qa, "prose": prose}
            if journal_schema == "eggroll-es-anchor-alpha-line-search-v4":
                for split in ("validation", "ood_qa"):
                    expected = V4_FROZEN_S6_BASELINE[split]
                    actual = normalized_qa[split]
                    _require(
                        actual["rows"] == expected["rows"]
                        and actual["exact"] == expected["exact"]
                        and actual["nonzero"] == expected["nonzero"]
                        and _same_float(
                            actual["mean_reward"], expected["mean_reward"],
                        ),
                        f"v4 alpha-zero {split} did not reproduce frozen S6",
                    )
                expected_prose = V4_FROZEN_S6_BASELINE["ood_prose"]
                _require(
                    prose["item_count"] == expected_prose["item_count"]
                    and prose["scored_token_count"]
                    == expected_prose["scored_token_count"]
                    and _same_float(
                        prose["mean_token_logprob"],
                        expected_prose["mean_token_logprob"],
                    ),
                    "v4 alpha-zero OOD prose did not reproduce frozen S6",
                )
        _require(
            normalized_qa["validation"]["rows"]
            == baseline["qa"]["validation"]["rows"],
            f"state {index} validation row count changed",
        )
        _require(
            normalized_qa["ood_qa"]["rows"]
            == baseline["qa"]["ood_qa"]["rows"],
            f"state {index} OOD-QA row count changed",
        )
        _require(
            prose["item_count"] == baseline["prose"]["item_count"]
            and prose["scored_token_count"]
            == baseline["prose"]["scored_token_count"],
            f"state {index} OOD-prose alignment counts changed",
        )
        qa_gate = _validate_qa_gate(
            state.get("ood_qa_gate"), baseline["qa"]["ood_qa"],
            normalized_qa["ood_qa"], f"state {index}",
        )
        prose_gate = _validate_prose_gate(
            state.get("ood_prose_gate"), baseline["prose"], prose,
            f"state {index}",
        )
        strict_passed = qa_gate["passed"] and prose_gate["passed"]
        _require(
            state.get("strict_guards_passed") is strict_passed,
            f"state {index} strict guard decision is inconsistent",
        )
        normalized_states.append({
            "state_index": index,
            "target_alpha": target,
            "alpha_increment": expected_increment,
            "qa": normalized_qa,
            "ood_qa_gate": qa_gate,
            "ood_prose": prose,
            "ood_prose_gate": prose_gate,
            "strict_guards_passed": strict_passed,
        })

    family_snapshot = copy.deepcopy(snapshot)
    family_snapshot.pop("fixed_train_batch")
    family_snapshot.pop("seed")
    family_snapshot.pop("targets")
    family_snapshot["recipe"].pop("seed", None)
    family_configuration = copy.deepcopy(configuration)
    family_configuration.pop("global_seed", None)
    validated = {
        "schema": journal_schema,
        "seed": snapshot["seed"],
        "targets": targets,
        "states": normalized_states,
        "snapshot": snapshot,
        "family_snapshot_sha256": canonical_sha256(family_snapshot),
        "family_configuration_sha256": canonical_sha256(family_configuration),
        "coefficient_sha256": coefficient_sha,
        "content_sha256": journal["content_sha256_before_self_field"],
    }
    if distributed_v3 is not None:
        validated["distributed_update_v3"] = distributed_v3
    if distributed_v4 is not None:
        validated["distributed_update_v4"] = distributed_v4
    return validated


def summarize_pilot(journal, source_path=None):
    validated = validate_journal(journal)
    baseline = validated["states"][0]
    states = []
    for state in validated["states"][1:]:
        validation_delta = (
            state["qa"]["validation"]["mean_reward"]
            - baseline["qa"]["validation"]["mean_reward"]
        )
        states.append({
            "target_alpha": state["target_alpha"],
            "path_dependent_monotonic_state": True,
            "validation_delta": validation_delta,
            "ood_qa_gate": state["ood_qa_gate"],
            "ood_prose_gate": state["ood_prose_gate"],
            "pilot_eligible": (
                validation_delta > 0.0
                and state["strict_guards_passed"]
            ),
        })
    report = {
        "schema": "eggroll-es-anchor-pilot-summary-v1",
        "classification": "exploratory_monotonic_pilot",
        "direct_confirmation": False,
        "selection_allowed": False,
        "source": str(source_path) if source_path is not None else None,
        "seed": validated["seed"],
        "targets": validated["targets"],
        "family_snapshot_sha256": validated["family_snapshot_sha256"],
        "family_configuration_sha256": validated["family_configuration_sha256"],
        "states": states,
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    return report


def aggregate_direct_confirmations(
    journals,
    *,
    candidate_name,
    expected_seeds=DEFAULT_CONFIRMATION_SEEDS,
    source_paths=None,
):
    _require(
        isinstance(candidate_name, str) and candidate_name.strip(),
        "candidate name is required",
    )
    _assert_no_heldout(candidate_name, "candidate name")
    expected_seeds = tuple(expected_seeds)
    _require(len(expected_seeds) == 5, "exactly five confirmation seeds are required")
    _require(len(set(expected_seeds)) == 5, "confirmation seeds must be unique")
    _require(len(journals) == 5, "exactly five journals are required")
    validated = [validate_journal(journal) for journal in journals]
    observed_seeds = [item["seed"] for item in validated]
    _require(
        set(observed_seeds) == set(expected_seeds)
        and len(set(observed_seeds)) == 5,
        "journal seeds do not match the five predeclared confirmation seeds",
    )
    validated.sort(key=lambda item: expected_seeds.index(item["seed"]))
    if source_paths is None:
        source_paths = [None] * len(validated)
    else:
        path_by_seed = {
            validate_journal(journal)["seed"]: str(path)
            for journal, path in zip(journals, source_paths)
        }
        source_paths = [path_by_seed[item["seed"]] for item in validated]

    for item in validated:
        _require(
            len(item["targets"]) == 2 and len(item["states"]) == 2,
            "monotonic pilot states cannot masquerade as direct confirmations",
        )
        alpha = item["targets"][1]
        _require(alpha > 0.0, "direct confirmation alpha must be positive")
        _require(
            _same_float(item["states"][1]["alpha_increment"], alpha),
            "confirmation was not a direct zero-to-alpha update",
        )

    family_snapshot_hashes = {
        item["family_snapshot_sha256"] for item in validated
    }
    family_configuration_hashes = {
        item["family_configuration_sha256"] for item in validated
    }
    _require(
        len(family_snapshot_hashes) == 1,
        "dataset, eval, implementation, or recipe identity differs across seeds",
    )
    _require(
        len(family_configuration_hashes) == 1,
        "trainer configuration differs across seeds",
    )
    alphas = {item["targets"][1] for item in validated}
    _require(len(alphas) == 1, "direct confirmation alpha differs across seeds")
    alpha = next(iter(alphas))

    per_seed = []
    validation_deltas = []
    for item, source_path in zip(validated, source_paths):
        baseline, treatment = item["states"]
        validation_delta = (
            treatment["qa"]["validation"]["mean_reward"]
            - baseline["qa"]["validation"]["mean_reward"]
        )
        validation_deltas.append(validation_delta)
        per_seed.append({
            "seed": item["seed"],
            "source": source_path,
            "journal_content_sha256": item["content_sha256"],
            "coefficient_sha256": item["coefficient_sha256"],
            "direct_alpha": alpha,
            "validation": {
                "baseline": baseline["qa"]["validation"]["mean_reward"],
                "treatment": treatment["qa"]["validation"]["mean_reward"],
                "delta": validation_delta,
                "positive": validation_delta > 0.0,
            },
            "ood_qa_gate": treatment["ood_qa_gate"],
            "ood_prose_gate": treatment["ood_prose_gate"],
            "all_strict_ood_guards_passed": treatment["strict_guards_passed"],
        })

    mean_delta = statistics.mean(validation_deltas)
    median_delta = statistics.median(validation_deltas)
    population_stddev = statistics.pstdev(validation_deltas)
    risk_adjusted = mean_delta - RISK_PENALTY * population_stddev
    positive_count = sum(delta > 0.0 for delta in validation_deltas)
    all_ood_passed = all(
        item["all_strict_ood_guards_passed"] for item in per_seed
    )
    eligible = (
        all_ood_passed
        and positive_count >= MIN_POSITIVE_SEEDS
        and mean_delta > 0.0
        and median_delta > 0.0
        and risk_adjusted > 0.0
    )
    report = {
        "schema": "eggroll-es-anchor-direct-confirmation-aggregate-v1",
        "classification": "direct_zero_to_alpha_five_seed_confirmation",
        "candidate_name": candidate_name,
        "direct_confirmation": True,
        "path_dependent_pilot_states_counted": False,
        "expected_seeds": list(expected_seeds),
        "direct_alpha": alpha,
        "family_snapshot_sha256": next(iter(family_snapshot_hashes)),
        "family_configuration_sha256": next(iter(family_configuration_hashes)),
        "policy": {
            "strict_ood_qa_mean_exact_nonzero_nondegradation": True,
            "strict_ood_prose_point_delta_minimum": 0.0,
            "strict_ood_prose_bootstrap_lower_bound_minimum": 0.0,
            "required_positive_validation_seeds": MIN_POSITIVE_SEEDS,
            "required_seed_count": 5,
            "mean_validation_delta_must_be_positive": True,
            "median_validation_delta_must_be_positive": True,
            "risk_penalty_population_stddev": RISK_PENALTY,
        },
        "per_seed": per_seed,
        "aggregate_validation": {
            "deltas": validation_deltas,
            "positive_seed_count": positive_count,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "population_stddev": population_stddev,
            "risk_adjusted_score": risk_adjusted,
        },
        "all_strict_ood_guards_passed": all_ood_passed,
        "eligible": eligible,
        "selected": candidate_name if eligible else "baseline",
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    return report


def atomic_write_json(path, value):
    path = Path(path)
    _assert_no_heldout(str(path), "output path")
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise JournalValidationError("output exists; overwrite and resume are forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("journals", nargs="+")
    parser.add_argument(
        "--mode", choices=("pilot", "direct-confirmation"), required=True,
    )
    parser.add_argument("--candidate-name")
    parser.add_argument("--expected-seeds", default="42,43,44,45,46")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    journals = [load_journal(path) for path in args.journals]
    if args.mode == "pilot":
        _require(len(journals) == 1, "pilot mode requires exactly one journal")
        report = summarize_pilot(journals[0], args.journals[0])
    else:
        _require(args.candidate_name is not None, "direct confirmation requires a candidate name")
        try:
            expected_seeds = tuple(
                int(piece.strip())
                for piece in args.expected_seeds.split(",")
                if piece.strip()
            )
        except ValueError as error:
            raise JournalValidationError("expected seeds must be integers") from error
        report = aggregate_direct_confirmations(
            journals,
            candidate_name=args.candidate_name,
            expected_seeds=expected_seeds,
            source_paths=args.journals,
        )
    atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
