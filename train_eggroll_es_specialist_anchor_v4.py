#!/usr/bin/env python3
"""Frozen-layer, dense-reward anchored EGGROLL-ES v4 recipe.

V4 retains v3's four-engine exact-reference and prose-anchor protocol while
pinning the perturbed parameter set to a pre-generated layer-plan artifact.
The plan is installed on every worker before v2/v3 capture the first exact
reference.  Domain fitness is a dense teacher-forced gold-answer objective;
validation remains the unchanged generated-answer QA evaluation.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import train_eggroll_es_specialist_anchor_v3 as anchor_v3


ROOT = Path(__file__).resolve().parent
REQUIRED_ENGINE_COUNT = anchor_v3.REQUIRED_ENGINE_COUNT
WORKER_EXTENSION = (
    "eggroll_es_worker_v4.FrozenLayerPlanAuditWorkerExtensionV4"
)
MAX_DENSE_REWARD_TOKENS = 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_LAYER_PLAN_BUNDLE = None
UPDATE_BINDING_KEYS_V4 = (
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
)
FROZEN_RUNTIME_EXPECTATIONS_V4 = {
    "6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb": {
        "source_unit_count": 70,
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "selected_byte_count": 571_998_208,
    },
    "b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0": {
        "source_unit_count": 70,
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "selected_byte_count": 571_998_208,
    },
}


def canonical_sha256(value):
    return anchor_v3.canonical_sha256_v3(value)


def coefficient_sha256(seeds, coefficients):
    return anchor_v3.coefficient_sha256(seeds, coefficients)


def file_sha256(path):
    return anchor_v3.file_sha256(path)


def load_anchor_prose(*args, **kwargs):
    return anchor_v3.load_anchor_prose(*args, **kwargs)


def dense_gold_reward_config_v4():
    """Return the immutable semantics whose hash is bound into every update."""
    return {
        "schema": "eggroll-es-dense-qa-reward-v1",
        "objective": "teacher_forced_gold_answer_prompt_logprob",
        "text_construction": "exact_prompt_plus_answer",
        "tokenization": {
            "add_special_tokens": False,
            "append_eos": False,
            "max_total_tokens": MAX_DENSE_REWARD_TOKENS,
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


DENSE_GOLD_REWARD_CONFIG_V4 = dense_gold_reward_config_v4()
DENSE_GOLD_REWARD_CONFIG_SHA256_V4 = canonical_sha256(
    DENSE_GOLD_REWARD_CONFIG_V4,
)


def _require_sha256(value, label):
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def load_frozen_layer_plan_v4(
    path,
    *,
    expected_file_sha256,
    expected_plan_sha256,
    expected_model_config_sha256,
):
    """Load one immutable es_layer_plan JSON and verify all external pins."""
    path = Path(path).resolve()
    expected_file_sha256 = _require_sha256(
        expected_file_sha256, "expected layer-plan file SHA256",
    )
    expected_plan_sha256 = _require_sha256(
        expected_plan_sha256, "expected layer-plan SHA256",
    )
    expected_model_config_sha256 = _require_sha256(
        expected_model_config_sha256, "expected model-config SHA256",
    )
    raw = path.read_bytes()
    actual_file_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError("frozen layer-plan file SHA256 changed")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("frozen layer-plan file is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("frozen layer-plan manifest is not an object")
    if manifest.get("schema") != "qwen36-es-layer-plan-v1":
        raise ValueError("frozen layer-plan schema changed")
    embedded_plan_sha256 = manifest.get("plan_sha256")
    _require_sha256(embedded_plan_sha256, "embedded layer-plan SHA256")
    unhashed = dict(manifest)
    unhashed.pop("plan_sha256", None)
    actual_plan_sha256 = canonical_sha256(unhashed)
    if (
        actual_plan_sha256 != embedded_plan_sha256
        or actual_plan_sha256 != expected_plan_sha256
    ):
        raise ValueError("frozen layer-plan canonical SHA256 changed")
    if (
        manifest.get("model_config_sha256")
        != expected_model_config_sha256
    ):
        raise ValueError("frozen layer-plan model-config SHA256 changed")
    model_config_path = Path(str(manifest.get("model_config", "")))
    if not model_config_path.is_absolute():
        model_config_path = (path.parent / model_config_path).resolve()
    if (
        not model_config_path.is_file()
        or file_sha256(model_config_path) != expected_model_config_sha256
    ):
        raise ValueError("referenced model config does not match its SHA256")
    units = manifest.get("units")
    layers = manifest.get("layers")
    include_regex = manifest.get("include_regex")
    if (
        not isinstance(units, list)
        or not units
        or not all(isinstance(item, str) and item for item in units)
        or len(set(units)) != len(units)
        or manifest.get("num_units") != len(units)
        or not isinstance(layers, list)
        or not layers
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in layers
        )
        or len(set(layers)) != len(layers)
        or not isinstance(include_regex, str)
        or not include_regex
    ):
        raise ValueError("frozen layer-plan target metadata is invalid")
    try:
        matcher = re.compile(include_regex)
    except re.error as error:
        raise ValueError("frozen layer-plan include regex is invalid") from error
    if any(matcher.fullmatch(unit) is None for unit in units):
        raise ValueError("frozen layer-plan regex does not match every unit")
    return {
        "schema": "eggroll-es-frozen-layer-plan-bundle-v4",
        "path": str(path),
        "file_sha256": actual_file_sha256,
        "plan_sha256": actual_plan_sha256,
        "model_config_path": str(model_config_path),
        "model_config_sha256": expected_model_config_sha256,
        "manifest": manifest,
    }


def parse_frozen_layer_plan_cli_v4(argv):
    """Consume the required path/hash quartet and return unrelated CLI args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--layer-plan-json")
    parser.add_argument("--expected-layer-plan-file-sha256")
    parser.add_argument("--expected-layer-plan-sha256")
    parser.add_argument("--expected-model-config-sha256")
    options, remaining = parser.parse_known_args(list(argv))
    values = (
        options.layer_plan_json,
        options.expected_layer_plan_file_sha256,
        options.expected_layer_plan_sha256,
        options.expected_model_config_sha256,
    )
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError(
            "layer-plan path and all three expected hashes must be paired"
        )
    if not all(value is not None for value in values):
        raise ValueError(
            "v4 requires --layer-plan-json and all expected layer-plan hashes"
        )
    bundle = load_frozen_layer_plan_v4(
        options.layer_plan_json,
        expected_file_sha256=options.expected_layer_plan_file_sha256,
        expected_plan_sha256=options.expected_layer_plan_sha256,
        expected_model_config_sha256=options.expected_model_config_sha256,
    )
    return bundle, remaining


def set_default_layer_plan_bundle_v4(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema")
        != "eggroll-es-frozen-layer-plan-bundle-v4"
    ):
        raise ValueError("default v4 layer-plan bundle is invalid")
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def _encode_without_special_tokens(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, (list, tuple)) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        raise ValueError("tokenizer returned invalid token IDs")
    return list(token_ids)


def prepare_gold_answer_items_v4(
    tokenizer, prompts, answers, max_total_tokens=MAX_DENSE_REWARD_TOKENS,
):
    """Tokenize exact prompt+answer strings and prove the answer boundary."""
    prompts = list(prompts)
    answers = list(answers)
    if len(prompts) == 0 or len(prompts) != len(answers):
        raise ValueError("gold reward prompt/answer counts differ or are empty")
    if max_total_tokens != MAX_DENSE_REWARD_TOKENS:
        raise ValueError("v4 dense reward token cap is frozen at 1024")
    items = []
    for index, (prompt, answer) in enumerate(zip(prompts, answers)):
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("gold reward prompts and answers must be strings")
        if not prompt or not answer:
            raise ValueError("gold reward prompts and answers must be non-empty")
        prompt_ids = _encode_without_special_tokens(tokenizer, prompt)
        combined_ids = _encode_without_special_tokens(tokenizer, prompt + answer)
        if not prompt_ids:
            raise ValueError("gold reward prompt tokenization is empty")
        if combined_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"gold reward tokenizer boundary mismatch at example {index}"
            )
        answer_ids = combined_ids[len(prompt_ids):]
        if not answer_ids:
            raise ValueError(
                f"gold reward answer has no aligned tokens at example {index}"
            )
        if len(combined_ids) > max_total_tokens:
            raise ValueError(
                f"gold reward example {index} has {len(combined_ids)} tokens, "
                "above the frozen 1024-token cap"
            )
        items.append({
            "example_index": index,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_ids),
            "answer_token_start": len(prompt_ids),
            "answer_token_count": len(answer_ids),
            "prompt_token_ids": combined_ids,
            "prompt_token_ids_sha256": canonical_sha256(combined_ids),
            "eos_appended": False,
        })
    return items


def score_gold_answer_outputs_v4(items, outputs):
    """Average answer-token logprobs per example, then average examples."""
    if len(items) != len(outputs) or not items:
        raise ValueError("gold reward item/output counts differ or are empty")
    examples = []
    for item, output in zip(items, outputs):
        expected_ids = item["prompt_token_ids"]
        returned_ids = list(getattr(output, "prompt_token_ids", None) or [])
        if returned_ids != expected_ids:
            raise ValueError(
                "vLLM truncated or changed the teacher-forced prompt token IDs"
            )
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None:
            raise ValueError("vLLM omitted teacher-forced prompt logprobs")
        if len(prompt_logprobs) != len(expected_ids):
            raise ValueError(
                "vLLM prompt-logprob length indicates truncation or mismatch"
            )
        start = item["answer_token_start"]
        stop = start + item["answer_token_count"]
        if start <= 0 or stop != len(expected_ids):
            raise ValueError("gold reward answer boundary metadata changed")
        values = []
        for position in range(start, stop):
            token_id = expected_ids[position]
            candidates = prompt_logprobs[position]
            if candidates is None or token_id not in candidates:
                raise ValueError(
                    f"vLLM omitted gold token {token_id} at position {position}"
                )
            selected = candidates[token_id]
            value = float(
                selected.logprob
                if hasattr(selected, "logprob") else selected["logprob"]
            )
            if not math.isfinite(value):
                raise ValueError("vLLM returned a non-finite gold logprob")
            values.append(value)
        token_sum = math.fsum(values)
        token_mean = token_sum / len(values)
        examples.append({
            "example_index": item["example_index"],
            "prompt_sha256": item["prompt_sha256"],
            "answer_sha256": item["answer_sha256"],
            "prompt_token_ids_sha256": item["prompt_token_ids_sha256"],
            "answer_token_count": len(values),
            "sum_answer_token_logprob": token_sum,
            "mean_answer_token_logprob": token_mean,
            "eos_scored": False,
        })
    example_means = [item["mean_answer_token_logprob"] for item in examples]
    return {
        "schema": "eggroll-es-dense-gold-reward-result-v4",
        "reward_config_sha256": DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
        "example_count": len(examples),
        "answer_token_count": sum(
            item["answer_token_count"] for item in examples
        ),
        "mean_example_mean_logprob": (
            math.fsum(example_means) / len(example_means)
        ),
        "examples": examples,
    }


def update_manifest_v4(
    *, coefficient_sha256, population_size, world_size,
    reference_generation, plan_id, update_sequence, previous_alpha,
    target_alpha, expected_base_sha256, bindings,
):
    """Independently reconstruct the controller/worker v4 contract."""
    if not isinstance(bindings, dict) or set(bindings) != set(
        UPDATE_BINDING_KEYS_V4
    ):
        raise ValueError("v4 update manifest binding fields changed")
    return {
        "schema": "eggroll-es-layer-restricted-update-manifest-v4",
        "coefficient_sha256": str(coefficient_sha256),
        "population_size": int(population_size),
        "world_size": int(world_size),
        "reference_generation": int(reference_generation),
        "plan_id": str(plan_id),
        "update_sequence": int(update_sequence),
        "previous_alpha": float(previous_alpha),
        "target_alpha": float(target_alpha),
        "expected_base_sha256": str(expected_base_sha256),
        **bindings,
    }


def validate_layer_plan_installations_v4(reports, bundle):
    """Require unanimous worker matching before any exact reference exists."""
    if len(reports) != REQUIRED_ENGINE_COUNT:
        raise RuntimeError("v4 layer-plan install did not cover four engines")
    expected_runtime = FROZEN_RUNTIME_EXPECTATIONS_V4.get(
        bundle.get("plan_sha256"),
    )
    if not isinstance(expected_runtime, dict):
        raise RuntimeError(
            "v4 controller has no frozen runtime expectation for this plan"
        )
    ranks = []
    bindings = []
    for report in reports:
        if (
            not isinstance(report, dict)
            or report.get("schema")
            != "eggroll-es-layer-plan-installed-v4"
            or report.get("installed") is not True
            or report.get("world_size") != REQUIRED_ENGINE_COUNT
            or report.get("layer_plan_sha256") != bundle["plan_sha256"]
            or report.get("layer_plan_file_sha256") != bundle["file_sha256"]
            or report.get("dense_reward_sha256")
            != DENSE_GOLD_REWARD_CONFIG_SHA256_V4
            or report.get("reference_present_before_install") is not False
            or report.get("reference_generation_before_install") != 0
        ):
            raise RuntimeError(
                "a v4 worker installed a different or late layer plan"
            )
        rank = report.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError("v4 layer-plan worker rank is invalid")
        ranks.append(rank)
        binding = {
            key: report.get(key)
            for key in (
                "layer_plan_file_sha256", "layer_plan_sha256",
                "checkpoint_to_runtime_mapping_sha256", "source_unit_count",
                "runtime_selected_name_sha256",
                "selected_parameter_manifest_sha256",
                "runtime_selected_parameter_count", "selected_element_count",
                "unselected_origin_sha256", "dense_reward_sha256",
            )
        }
        for key in (
            "checkpoint_to_runtime_mapping_sha256",
            "runtime_selected_name_sha256",
            "selected_parameter_manifest_sha256",
            "unselected_origin_sha256",
        ):
            _require_sha256(binding[key], key)
        if (
            binding["source_unit_count"]
            != expected_runtime["source_unit_count"]
            or binding["source_unit_count"]
            != bundle["manifest"]["num_units"]
            or binding["runtime_selected_parameter_count"]
            != expected_runtime["runtime_selected_parameter_count"]
            or binding["selected_element_count"]
            != expected_runtime["selected_element_count"]
            or report.get("selected_byte_count")
            != expected_runtime["selected_byte_count"]
        ):
            raise RuntimeError("v4 runtime layer-plan mapping counts are invalid")
        initial_selected = report.get("initial_identity", {}).get("selected")
        if (
            not isinstance(initial_selected, dict)
            or initial_selected.get("parameter_count")
            != expected_runtime["runtime_selected_parameter_count"]
            or initial_selected.get("total_elements")
            != expected_runtime["selected_element_count"]
            or initial_selected.get("total_bytes")
            != expected_runtime["selected_byte_count"]
            or initial_selected.get("total_bytes")
            != 2 * initial_selected.get("total_elements", -1)
        ):
            raise RuntimeError("v4 selected BF16 byte manifest is invalid")
        bindings.append(binding)
    if sorted(ranks) != list(range(REQUIRED_ENGINE_COUNT)):
        raise RuntimeError("v4 layer-plan install ranks are incomplete")
    if len({canonical_sha256(item) for item in bindings}) != 1:
        raise RuntimeError("v4 workers resolved different runtime layer mappings")
    return bindings[0]


def _validate_bound_v4_reports(reports, manifest_sha, bindings, executed=False):
    expected_schema = (
        "eggroll-es-layer-restricted-update-executed-v4"
        if executed else "eggroll-es-layer-restricted-update-prepared-v4"
    )
    for report in reports:
        if (
            not isinstance(report, dict)
            or report.get("schema") != expected_schema
            or report.get("manifest_sha256") != manifest_sha
            or any(report.get(key) != value for key, value in bindings.items())
        ):
            raise RuntimeError("v4 worker report provenance differs")


def validate_prepared_shards_v4(
    reports, seeds, coefficients, manifest_sha, reference_generation,
    expected_base_sha, update_sequence, bindings,
):
    """Validate selected-only preparation without accepting v3 preflight."""
    _validate_bound_v4_reports(reports, manifest_sha, bindings)
    if len(reports) != REQUIRED_ENGINE_COUNT:
        raise RuntimeError("v4 prepared update did not cover four engines")
    ranks = []
    all_indices = []
    expected_per_rank = len(seeds) // REQUIRED_ENGINE_COUNT
    for report in reports:
        if (
            report.get("prepared") is not True
            or report.get("world_size") != REQUIRED_ENGINE_COUNT
            or report.get("reference_generation") != reference_generation
            or report.get("base_sha256") != expected_base_sha
            or report.get("update_sequence") != update_sequence
        ):
            raise RuntimeError("v4 prepared shard state differs")
        rank = report.get("rank")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= REQUIRED_ENGINE_COUNT
        ):
            raise RuntimeError("v4 prepared shard rank is invalid")
        shard = anchor_v3.seed_shard_v3(
            seeds, coefficients, rank, REQUIRED_ENGINE_COUNT,
        )
        if (
            report.get("shard_indices") != shard["indices"]
            or report.get("shard_seeds") != shard["seeds"]
            or len(shard["indices"]) != expected_per_rank
            or report.get("shard_pair_sha256") != canonical_sha256({
                "seeds": shard["seeds"],
                "coefficients": shard["coefficients"],
            })
        ):
            raise RuntimeError("v4 prepared seed/coefficient shard differs")
        preflight = report.get("allocation_preflight")
        if (
            not isinstance(preflight, dict)
            or preflight.get("schema")
            != "eggroll-es-selected-allocation-preflight-v4"
            or preflight.get("passed") is not True
            or preflight.get("parameter_count")
            != bindings["runtime_selected_parameter_count"]
            or preflight.get("element_count")
            != bindings["selected_element_count"]
            or preflight.get("accumulator_dtype") != "torch.float32"
            or preflight.get("scratch_freed_before_collectives") is not True
            or preflight.get("collectives_created") is not False
            or preflight.get("rng_consumed") is not False
            or preflight.get("weights_changed") is not False
            or any(
                preflight.get(key) != value
                for key, value in bindings.items()
            )
        ):
            raise RuntimeError("v4 selected allocation preflight is unsafe")
        ranks.append(rank)
        all_indices.extend(shard["indices"])
    if sorted(ranks) != list(range(REQUIRED_ENGINE_COUNT)):
        raise RuntimeError("v4 prepared ranks are incomplete")
    if (
        sorted(all_indices) != list(range(len(seeds)))
        or len(set(all_indices)) != len(all_indices)
    ):
        raise RuntimeError("v4 prepared shards overlap or omit a seed")
    return True


class FrozenLayerDenseRewardMixinV4(
    anchor_v3.DistributedAnchoredStepMixinV3,
):
    """Install a frozen parameter plan and add dense gold-answer fitness."""

    def _rpc_all_engines_v4(self, method, args):
        handles = [
            engine.collective_rpc.remote(method, args=args)
            for engine in self.engines
        ]
        results = self._resolve(handles)
        return anchor_v3._unwrap_tp1_results_v3(
            results, len(self.engines), method,
        )

    def _persist_anchor_plan(self, plan):
        """Never expose coefficient artifacts before the boundary audit."""
        if (
            getattr(self, "_v4_withhold_unbound_plan", False)
            and not isinstance(
                plan.get("population_boundary_audit_v4"), dict,
            )
        ):
            if getattr(self, "_pending_identity_audit", None) is not None:
                plan["identity_audit"] = self._pending_identity_audit
            return None
        return super()._persist_anchor_plan(plan)

    def configure_anchor(self, *args, frozen_layer_plan=None, **kwargs):
        bundle = frozen_layer_plan or _DEFAULT_LAYER_PLAN_BUNDLE
        if (
            not isinstance(bundle, dict)
            or bundle.get("schema")
            != "eggroll-es-frozen-layer-plan-bundle-v4"
        ):
            raise ValueError("anchored v4 requires a validated frozen layer plan")
        if len(self.engines) != REQUIRED_ENGINE_COUNT:
            raise ValueError("anchored v4 requires exactly four engines")
        if int(self.n_vllm_engines) != REQUIRED_ENGINE_COUNT:
            raise ValueError("anchored v4 engine configuration changed")
        if int(self.n_gpu_per_vllm_engine) != 1:
            raise ValueError("anchored v4 requires TP=1")
        if self.population_size % REQUIRED_ENGINE_COUNT != 0:
            raise ValueError(
                "anchored v4 population must be divisible by four engines"
            )
        reports = self._rpc_all_engines_v4(
            "install_layer_plan_v4",
            (
                Path(bundle["path"]).read_bytes(),
                bundle["file_sha256"],
                bundle["plan_sha256"],
                DENSE_GOLD_REWARD_CONFIG_V4,
                DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            ),
        )
        install = validate_layer_plan_installations_v4(reports, bundle)
        self._v4_layer_plan = bundle
        self._v4_layer_plan_install = install
        self._v4_reward_config = dict(DENSE_GOLD_REWARD_CONFIG_V4)
        self._v4_reward_config_sha256 = DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        # Invoke v2's inherited exact-reference setup directly: v4 forbids the
        # v3 state RPC after plan installation, but retains v2's selected-only
        # save/restore API.  This call must remain after the install RPC.
        result = (
            anchor_v3.anchor_v2.ExactRestoredAnchoredStepMixin
            .configure_anchor(self, *args, **kwargs)
        )
        states = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4",
            (REQUIRED_ENGINE_COUNT, "exact_reference"),
        )
        summary = self._validate_worker_states_v4(
            states, require_fresh=True,
        )
        self._set_coordinator_reference_v3(summary, fresh=True)
        return result

    def _validate_worker_states_v4(self, states, require_fresh):
        summary = self._validate_worker_states_v3(states, require_fresh)
        for state in states:
            if any(
                state.get(key) != value
                for key, value in self._v4_layer_plan_install.items()
            ):
                raise RuntimeError("v4 worker state lost its frozen layer plan")
        return summary

    def _refresh_population_references_v4(self):
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v4 workers captured different exact references")
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4",
            (REQUIRED_ENGINE_COUNT, "exact_reference"),
        )
        summary = self._validate_worker_states_v4(
            inspected, require_fresh=True,
        )
        self._set_coordinator_reference_v3(summary, fresh=True)
        self._exact_reference_states = [[state] for state in states]

    def _dense_sampling_params_v4(self, iteration):
        return self._sampling_params(
            n=1,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )

    def _population_boundary_audit_v4(self, iteration):
        """Hash both partitions once after population restore, before update."""
        reports = self._rpc_all_engines_v4(
            "audit_population_completion_v4",
            (
                REQUIRED_ENGINE_COUNT,
                self._v3_reference_generation,
                self._v3_reference_identity["sha256"],
            ),
        )
        if len(reports) != REQUIRED_ENGINE_COUNT:
            raise RuntimeError(
                "v4 post-population audit did not cover four engines"
            )
        ranks = []
        for report in reports:
            if (
                not isinstance(report, dict)
                or report.get("schema")
                != "eggroll-es-post-population-audit-v4"
                or report.get("passed") is not True
                or report.get("world_size") != REQUIRED_ENGINE_COUNT
                or report.get("reference_generation")
                != self._v3_reference_generation
                or report.get("reference_sha256")
                != self._v3_reference_identity["sha256"]
                or report.get("current_identity")
                != self._v3_reference_identity
                or any(
                    report.get(key) != value
                    for key, value in self._v4_layer_plan_install.items()
                )
            ):
                raise RuntimeError(
                    "a v4 post-population worker audit differs"
                )
            ranks.append(report.get("rank"))
        if sorted(ranks) != list(range(REQUIRED_ENGINE_COUNT)):
            raise RuntimeError("v4 post-population audit ranks are incomplete")
        if len({
            canonical_sha256(report["current_identity"])
            for report in reports
        }) != 1:
            raise RuntimeError("v4 post-population model identities differ")
        audit = {
            "schema": "eggroll-es-population-boundary-audit-v4",
            "iteration": int(iteration),
            "phase": "after_complete_population_exact_restore_before_plan",
            "engine_count": REQUIRED_ENGINE_COUNT,
            "reference_generation": self._v3_reference_generation,
            "reference_identity": dict(self._v3_reference_identity),
            "current_identity": dict(self._v3_reference_identity),
            "unselected_origin_sha256": self._v4_layer_plan_install[
                "unselected_origin_sha256"
            ],
            "runtime_mapping": dict(self._v4_layer_plan_install),
            "worker_reports": reports,
            "passed": True,
        }
        audit["audit_sha256"] = canonical_sha256(audit)
        return audit

    def _evaluate_population_with_anchor(
        self, seeds, input_batch, target_batch, domain_sampling_params,
        anchor_items, iteration,
    ):
        del domain_sampling_params
        dense_items = prepare_gold_answer_items_v4(
            self.tokenizer, input_batch, target_batch,
        )
        dense_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in dense_items
        ]
        dense_sampling = self._dense_sampling_params_v4(iteration)
        anchor_sampling = self._sampling_params(
            n=1,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )
        anchor_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in anchor_items
        ]
        seeds_perf = {}
        anchor_scores = {}
        results = []
        for start in range(0, len(seeds), len(self.engines)):
            engine_batch = seeds[start:start + len(self.engines)]
            if len(engine_batch) != len(self.engines):
                raise ValueError("partial v4 population wave would idle a GPU")
            dense_batches = None
            anchor_batches = None
            try:
                self._resolve([
                    self.engines[index].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(int(seed), self.sigma, False),
                    )
                    for index, seed in enumerate(engine_batch)
                ])
                dense_batches = self._resolve([
                    self.engines[index].generate.remote(
                        list(dense_prompts), dense_sampling, use_tqdm=False,
                    )
                    for index, _ in enumerate(engine_batch)
                ])
                if anchor_items:
                    anchor_batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(anchor_prompts), anchor_sampling,
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(engine_batch)
                    ])
            finally:
                self._restore_all_engines_exact()
            if len(dense_batches) != len(engine_batch):
                raise ValueError("v4 dense population engine count changed")
            if anchor_items and len(anchor_batches) != len(engine_batch):
                raise ValueError("v4 anchor population engine count changed")
            for index, seed in enumerate(engine_batch):
                dense = score_gold_answer_outputs_v4(
                    dense_items, dense_batches[index],
                )
                example_rewards = [
                    item["mean_answer_token_logprob"]
                    for item in dense["examples"]
                ]
                seeds_perf[int(seed)] = {
                    "avg_reward": dense["mean_example_mean_logprob"],
                    "rewards": example_rewards,
                    "results": dense["examples"],
                    "dense_gold_reward_v4": dense,
                }
                results.append({
                    "seed": int(seed),
                    "avg_reward": dense["mean_example_mean_logprob"],
                })
                if anchor_items:
                    outputs = anchor_batches[index]
                    if len(outputs) != len(anchor_items):
                        raise ValueError(
                            "v4 population engine changed anchor request count"
                        )
                    anchor_scores[int(seed)] = (
                        anchor_v3.anchor_v2.anchor_v1.score_anchor_outputs(
                            anchor_items, outputs,
                        )
                    )
        return seeds_perf, anchor_scores, results

    def _identity_probe(
        self, input_batch, domain_sampling, anchor_items, iteration,
    ):
        del domain_sampling
        probe_count = int(getattr(self, "_v4_identity_probe_count", 0)) + 1
        self._v4_identity_probe_count = probe_count
        if probe_count == 2:
            self._v4_pending_population_boundary_audit = (
                self._population_boundary_audit_v4(iteration)
            )
        elif probe_count > 2:
            raise RuntimeError("v4 identity audit performed an extra probe")
        answers = getattr(self, "_v4_identity_target_answers", None)
        if answers is None:
            raise RuntimeError("v4 identity probe has no gold answers")
        dense_items = prepare_gold_answer_items_v4(
            self.tokenizer, input_batch, answers,
        )
        if min(len(dense_items), len(self.engines)) != REQUIRED_ENGINE_COUNT:
            raise RuntimeError(
                "v4 identity probe domain batch does not cover all four engines"
            )
        dense_outputs = anchor_v3.anchor_v2.anchor_v1.dispatch_eval_batch(
            self.engines,
            [{"prompt_token_ids": item["prompt_token_ids"]}
             for item in dense_items],
            self._dense_sampling_params_v4(iteration),
            self._resolve,
        )
        dense = score_gold_answer_outputs_v4(dense_items, dense_outputs)
        anchor_sampling = self._sampling_params(
            n=1,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )
        anchor_outputs = anchor_v3.anchor_v2.anchor_v1.dispatch_eval_batch(
            self.engines,
            [{"prompt_token_ids": item["prompt_token_ids"]}
             for item in anchor_items],
            anchor_sampling,
            self._resolve,
        )
        return {
            "schema": "eggroll-es-train-only-identity-probe-v4",
            "dense_gold_output_sha256": canonical_sha256(dense),
            "anchor_output_sha256": anchor_v3.anchor_v2.anchor_output_sha256(
                anchor_items, anchor_outputs,
            ),
            "domain_requests": len(dense_items),
            "anchor_requests": len(anchor_items),
            "reward_config_sha256": self._v4_reward_config_sha256,
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "dispatch": "strided_engine_shards_separate_calls",
        }

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        if not getattr(self, "_v3_reference_fresh", False):
            self._refresh_population_references_v4()
        batches = self._iter_minibatches(
            input_text, target_text, self.mini_batch_size,
        )
        try:
            probe_inputs, probe_answers = next(batches)
        except StopIteration as error:
            raise ValueError("v4 identity audit received an empty batch") from error
        if not probe_inputs:
            raise ValueError("v4 identity audit received an empty batch")
        self._v4_identity_target_answers = list(probe_answers)
        self._v4_identity_probe_count = 0
        self._v4_pending_population_boundary_audit = None
        self._v4_withhold_unbound_plan = True
        try:
            plan = (
                anchor_v3.anchor_v2.ExactRestoredAnchoredStepMixin
                .estimate_step_coefficients(
                    self, iteration, seeds, input_text, target_text,
                )
            )
        finally:
            self._v4_identity_target_answers = None
            self._v4_withhold_unbound_plan = False
        boundary_audit = self._v4_pending_population_boundary_audit
        if (
            self._v4_identity_probe_count != 2
            or not isinstance(boundary_audit, dict)
            or boundary_audit.get("passed") is not True
            or boundary_audit.get("audit_sha256")
            != canonical_sha256({
                key: value for key, value in boundary_audit.items()
                if key != "audit_sha256"
            })
        ):
            raise RuntimeError("v4 population-boundary audit is missing or invalid")
        plan_id = canonical_sha256({
            "schema": "eggroll-es-distributed-plan-id-v4",
            "iteration": int(iteration),
            "coefficient_sha256": plan["coefficient_sha256"],
            "reference_generation": self._v3_reference_generation,
            "reference_sha256": self._v3_reference_identity["sha256"],
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "reward_config_sha256": self._v4_reward_config_sha256,
            "runtime_mapping_sha256": canonical_sha256(
                self._v4_layer_plan_install,
            ),
            "population_boundary_audit_sha256": boundary_audit[
                "audit_sha256"
            ],
        })
        plan["population_boundary_audit_v4"] = boundary_audit
        plan["dense_gold_reward_v4"] = {
            "config": dict(self._v4_reward_config),
            "reward_config_sha256": self._v4_reward_config_sha256,
        }
        plan["frozen_layer_plan_v4"] = {
            key: self._v4_layer_plan[key]
            for key in (
                "path", "file_sha256", "plan_sha256",
                "model_config_path", "model_config_sha256",
            )
        }
        plan["frozen_layer_plan_v4"]["runtime_mapping"] = dict(
            self._v4_layer_plan_install,
        )
        plan["frozen_layer_plan_v4"]["runtime_mapping_sha256"] = (
            canonical_sha256(self._v4_layer_plan_install)
        )
        plan["distributed_update_v4"] = {
            "schema": "eggroll-es-distributed-seed-plan-v4",
            "plan_id": plan_id,
            "engine_count": REQUIRED_ENGINE_COUNT,
            "tp_per_engine": 1,
            "reference_generation": self._v3_reference_generation,
            "reference_identity": dict(self._v3_reference_identity),
            "seed_sharding": "strided_by_inter_engine_rank",
            "collective_dtype": "torch.float32",
            "reference_recapture_policy": "once_before_next_population_only",
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "dense_reward_sha256": self._v4_reward_config_sha256,
            "runtime_mapping": dict(self._v4_layer_plan_install),
            "runtime_mapping_sha256": canonical_sha256(
                self._v4_layer_plan_install,
            ),
            "population_boundary_audit_sha256": boundary_audit[
                "audit_sha256"
            ],
        }
        self._v3_active_plan_id = plan_id
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._persist_anchor_plan(plan)
        return plan

    def _abort_update_v4(self, plan, failure):
        metadata = plan["distributed_update_v4"]
        aborts = self._rpc_all_engines_v4(
            "abort_distributed_update_v4",
            (metadata["plan_id"], metadata["reference_generation"]),
        )
        if len(aborts) != REQUIRED_ENGINE_COUNT or any(
            not isinstance(item, dict)
            or item.get("schema")
            != "eggroll-es-layer-restricted-update-abort-v4"
            or item.get("aborted") is not True
            or any(
                item.get(key) != value
                for key, value in self._v4_layer_plan_install.items()
            )
            for item in aborts
        ):
            raise RuntimeError("v4 exact-reference abort was not unanimous")
        identities = [item.get("restored_identity") for item in aborts]
        if len({canonical_sha256(item) for item in identities}) != 1:
            raise RuntimeError("v4 abort restored different worker weights")
        if identities[0] != metadata["reference_identity"]:
            raise RuntimeError("v4 abort did not restore retained reference")
        metadata["last_failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
            "aborted_to_reference": True,
            "restored_identity": identities[0],
        }
        plan["applied_alpha"] = 0.0
        self._v3_current_identity = dict(identities[0])
        self._v3_reference_fresh = True
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_active_plan_id = None
        self._persist_anchor_plan(plan)

    def apply_seed_coefficients(self, plan, target_alpha):
        audit = plan.get("identity_audit")
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise RuntimeError("anchored v4 requires a passed identity audit")
        if plan is not self._latest_anchor_plan:
            raise ValueError("only the latest v4 seed plan may be used")
        seeds, coefficients = anchor_v3.validate_seed_coefficients_v3(
            plan.get("seeds", []), plan.get("coefficients", []),
            self.population_size, REQUIRED_ENGINE_COUNT,
        )
        coefficient_identity = anchor_v3.coefficient_sha256_v3(
            seeds, coefficients,
        )
        if (
            coefficient_identity != plan.get("coefficient_sha256")
            or coefficient_identity != coefficient_sha256(seeds, coefficients)
        ):
            raise ValueError("v4 seed plan coefficient integrity failed")
        metadata = plan.get("distributed_update_v4")
        boundary_audit = plan.get("population_boundary_audit_v4")
        if (
            not isinstance(metadata, dict)
            or metadata.get("plan_id") != self._v3_active_plan_id
            or metadata.get("reference_generation")
            != self._v3_reference_generation
            or metadata.get("layer_plan_sha256")
            != self._v4_layer_plan["plan_sha256"]
            or metadata.get("dense_reward_sha256")
            != self._v4_reward_config_sha256
            or metadata.get("runtime_mapping")
            != self._v4_layer_plan_install
            or metadata.get("runtime_mapping_sha256")
            != canonical_sha256(self._v4_layer_plan_install)
            or not isinstance(boundary_audit, dict)
            or boundary_audit.get("passed") is not True
            or boundary_audit.get("audit_sha256")
            != canonical_sha256({
                key: value for key, value in boundary_audit.items()
                if key != "audit_sha256"
            })
            or metadata.get("population_boundary_audit_sha256")
            != boundary_audit.get("audit_sha256")
        ):
            raise RuntimeError("v4 distributed seed plan is stale or mismatched")
        target_alpha = float(target_alpha)
        previous_alpha = float(plan["applied_alpha"])
        if not math.isfinite(target_alpha) or target_alpha < previous_alpha:
            raise ValueError("target alpha must be finite and monotonic")
        if target_alpha == previous_alpha:
            return plan
        if previous_alpha != self._v3_accepted_alpha:
            raise RuntimeError("v4 coordinator alpha state is stale")
        if self._v3_update_sequence == 0 and not self._v3_reference_fresh:
            raise RuntimeError("first v4 update has a stale reference")
        if self._v3_update_sequence > 0 and self._v3_reference_fresh:
            raise RuntimeError("continued v4 update has a fresh reference")
        update_sequence = self._v3_update_sequence + 1
        expected_base_sha = self._v3_current_identity.get("sha256")
        expected_manifest = update_manifest_v4(
            coefficient_sha256=coefficient_identity,
            population_size=self.population_size,
            world_size=REQUIRED_ENGINE_COUNT,
            reference_generation=self._v3_reference_generation,
            plan_id=metadata["plan_id"],
            update_sequence=update_sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=expected_base_sha,
            bindings=self._v4_layer_plan_install,
        )
        manifest_sha = canonical_sha256(expected_manifest)
        try:
            prepared = self._rpc_all_engines_v4(
                "prepare_sharded_seed_update_v4",
                (
                    seeds, coefficients, coefficient_identity,
                    self.population_size, REQUIRED_ENGINE_COUNT,
                    self._v3_reference_generation, metadata["plan_id"],
                    update_sequence, previous_alpha, target_alpha,
                    expected_base_sha,
                    *(self._v4_layer_plan_install[key]
                      for key in UPDATE_BINDING_KEYS_V4),
                ),
            )
            _validate_bound_v4_reports(
                prepared, manifest_sha, self._v4_layer_plan_install,
            )
            validate_prepared_shards_v4(
                prepared, seeds, coefficients, manifest_sha,
                self._v3_reference_generation, expected_base_sha,
                update_sequence, self._v4_layer_plan_install,
            )
            executed = self._rpc_all_engines_v4(
                "execute_prepared_seed_update_v4", (manifest_sha,),
            )
            _validate_bound_v4_reports(
                executed, manifest_sha, self._v4_layer_plan_install,
                executed=True,
            )
            final_identity = anchor_v3.validate_executed_updates_v3(
                executed, manifest_sha,
            )
            committed = self._rpc_all_engines_v4(
                "commit_prepared_seed_update_v4",
                (manifest_sha, final_identity["sha256"]),
            )
            if (
                len(committed) != REQUIRED_ENGINE_COUNT
                or sorted(item.get("rank") for item in committed)
                != list(range(REQUIRED_ENGINE_COUNT))
                or any(
                    not isinstance(item, dict)
                    or item.get("schema")
                    != "eggroll-es-layer-restricted-update-committed-v4"
                    or item.get("committed") is not True
                    or item.get("manifest_sha256") != manifest_sha
                    or item.get("final_sha256") != final_identity["sha256"]
                    or item.get("reference_fresh_for_population") is not False
                    or item.get("update_sequence") != update_sequence
                    or any(
                        item.get(key) != value
                        for key, value in self._v4_layer_plan_install.items()
                    )
                    for item in committed
                )
            ):
                raise RuntimeError("v4 distributed commit was not unanimous")
            post_states = self._rpc_all_engines_v4(
                "inspect_cached_distributed_update_state_v4",
                (
                    REQUIRED_ENGINE_COUNT,
                    f"update_commit:{manifest_sha}",
                ),
            )
            post = self._validate_worker_states_v4(
                post_states, require_fresh=False,
            )
            if (
                post["reference_generation"] != self._v3_reference_generation
                or post["current_identity"] != final_identity
                or any(
                    state.get("update_session") != metadata["plan_id"]
                    or state.get("update_sequence") != update_sequence
                    or state.get("accepted_alpha") != target_alpha
                    for state in post_states
                )
            ):
                raise RuntimeError("post-commit v4 worker state differs")
        except Exception as error:
            try:
                self._abort_update_v4(plan, error)
            except Exception as abort_error:
                raise RuntimeError(
                    f"v4 update failed ({error}); exact abort also failed "
                    f"({abort_error})"
                ) from error
            raise
        self._v3_current_identity = dict(final_identity)
        self._v3_reference_fresh = False
        self._v3_update_sequence = update_sequence
        self._v3_accepted_alpha = target_alpha
        plan["applied_alpha"] = target_alpha
        plan["applications"].append({
            "schema": "eggroll-es-distributed-alpha-application-v4",
            "target_alpha": target_alpha,
            "alpha_increment": target_alpha - previous_alpha,
            "update_sequence": update_sequence,
            "manifest_sha256": manifest_sha,
            "manifest": expected_manifest,
            "coefficient_sha256": coefficient_identity,
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "dense_reward_sha256": self._v4_reward_config_sha256,
            "runtime_mapping": dict(self._v4_layer_plan_install),
            "prepared_shards": prepared,
            "executed_collectives": executed,
            "commits": committed,
            "post_commit_states": post_states,
            "final_identity": final_identity,
            "reference_recaptured": False,
            "reference_fresh_for_population": False,
            "bf16_alpha_semantics": "path_dependent_monotonic_increment",
            "direct_alpha_confirmation_required": True,
        })
        self._persist_anchor_plan(plan)
        return plan


def load_trainer(layer_plan_bundle=None):
    """Load the v3 coordinator with a v4 frozen-plan worker extension."""
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parent = anchor_v3.load_trainer()
    captured_bundle = layer_plan_bundle

    class FrozenLayerDenseAnchoredTrainerV4(
        FrozenLayerDenseRewardMixinV4, parent,
    ):
        def configure_anchor(self, *args, frozen_layer_plan=None, **kwargs):
            return super().configure_anchor(
                *args,
                frozen_layer_plan=(frozen_layer_plan or captured_bundle),
                **kwargs,
            )

        def launch_engines(
            self,
            num_engines=4,
            n_gpu_per_vllm_engine=1,
            model_name="Qwen/Qwen2.5-Math-1.5B",
            precision="bfloat16",
        ):
            if num_engines != REQUIRED_ENGINE_COUNT:
                raise ValueError("anchored v4 requires exactly four engines")
            if n_gpu_per_vllm_engine != 1:
                raise ValueError("anchored v4 requires TP=1")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )
            from es_at_scale.trainer.es_trainer import ESNcclLLM

            pgs = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}],
                    strategy="PACK",
                    lifetime="detached",
                )
                for _ in range(num_engines)
            ]
            ray.get([pg.ready() for pg in pgs])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for pg in pgs
            ]
            engine_args = {
                "model": model_name,
                "tensor_parallel_size": 1,
                "worker_extension_cls": WORKER_EXTENSION,
                "dtype": precision,
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.82,
                "max_model_len": 2048,
                "limit_mm_per_prompt": {"image": 0, "video": 0},
                "mm_processor_cache_gb": 0,
                "skip_mm_profiling": True,
                "moe_backend": "triton",
            }
            engines = [
                ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=strategy,
                )(ESNcclLLM).remote(**engine_args)
                for strategy in strategies
            ]
            return engines, pgs

    return FrozenLayerDenseAnchoredTrainerV4


def run_exact_steps(trainer, *args, **kwargs):
    summary = anchor_v3.run_exact_steps(trainer, *args, **kwargs)
    summary["schema"] = "eggroll-es-anchored-exact-run-v4"
    summary["anchor"]["distributed_update_v4"] = {
        "engine_count": REQUIRED_ENGINE_COUNT,
        "tp_per_engine": 1,
        "seed_sharding": "strided_by_inter_engine_rank",
        "collective_dtype": "torch.float32",
        "two_phase_commit": True,
        "final_hash_consensus_required": True,
        "reference_recapture_policy": "once_before_next_population_only",
        "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
        "direct_alpha_confirmation_required": True,
        "frozen_layer_plan": trainer._v4_layer_plan,
        "layer_plan_install": trainer._v4_layer_plan_install,
        "dense_gold_reward": {
            "config": trainer._v4_reward_config,
            "reward_config_sha256": trainer._v4_reward_config_sha256,
        },
    }
    summary["implementation"]["distributed_anchor_trainer_v4"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
    }
    worker_path = ROOT / "eggroll_es_worker_v4.py"
    summary["implementation"]["distributed_worker_extension_v4"] = {
        "path": str(worker_path.resolve()),
        "sha256": file_sha256(worker_path),
    }
    anchor_v3.anchor_v2._atomic_write_json(
        Path(trainer.logging_dir) / "run_summary.json", summary,
    )
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = parse_frozen_layer_plan_cli_v4(argv)
    set_default_layer_plan_bundle_v4(bundle)
    anchor_v1 = anchor_v3.anchor_v2.anchor_v1
    old_argv = sys.argv
    old_load = anchor_v1.load_trainer
    old_run = anchor_v1.run_exact_steps
    sys.argv = [old_argv[0], *remaining]
    anchor_v1.load_trainer = load_trainer
    anchor_v1.run_exact_steps = run_exact_steps
    try:
        anchor_v1.main()
    finally:
        sys.argv = old_argv
        anchor_v1.load_trainer = old_load
        anchor_v1.run_exact_steps = old_run


if __name__ == "__main__":
    main()
