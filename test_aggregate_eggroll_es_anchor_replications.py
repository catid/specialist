import copy
import hashlib

import pytest

from aggregate_eggroll_es_anchor_replications import (
    JournalValidationError,
    _validate_v4_bindings,
    aggregate_direct_confirmations,
    canonical_sha256,
    summarize_pilot,
    validate_journal,
)


V4_PLAN_SHA = (
    "6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb"
)
V4_PLAN_FILE_SHA = (
    "8e855cbd0d6130278e87b1af348e39dd0f683b8575d9abcb9260f3fe7b29d824"
)
V4_MODEL_CONFIG_SHA = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
V4_MAPPING_SHA = (
    "0a1b84e8ed53ef56c174e7fcac728a4820293505647ab6b9ea02bc86a012b3b1"
)
V4_RUNTIME_NAMES_SHA = (
    "417b3867ba9a56f909d01b1e7bb0b8bb04f903c3ec49438a6675239a7bab270f"
)
V4_MIDDLE_PLAN_SHA = (
    "b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0"
)
V4_MIDDLE_PLAN_FILE_SHA = (
    "f2b38054e3cdaf41619cce579d3ba2e030fa3cfa87fd42b50543f655ff5f6dc0"
)
V4_MIDDLE_MAPPING_SHA = (
    "d6f43de81bb5c41318a38f077b8a3e6272676801752ff68d4772977ac72182f7"
)
V4_MIDDLE_RUNTIME_NAMES_SHA = (
    "a7df9257f81c05a3fb3e858209486bd930aad0ddb94d7398e1644b779fb8b70d"
)
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
V4_DENSE_REWARD_SHA = (
    "4941f2e94091b1f8e7ab7b5294ebc6520b80aba1326b7dc6ccea5140a3da5da2"
)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def qa_summary(label, mean, exact, nonzero, rows):
    return {
        "path": f"/safe/{label}.json",
        "sha256": digest(label),
        "rows": rows,
        "mean_reward": mean,
        "exact": exact,
        "nonzero": nonzero,
    }


def qa_gate(baseline, candidate):
    deltas = {
        "mean_reward": candidate["mean_reward"] - baseline["mean_reward"],
        "exact": candidate["exact"] - baseline["exact"],
        "nonzero": candidate["nonzero"] - baseline["nonzero"],
    }
    return {
        "schema": "eggroll-es-strict-qa-nondegradation-v1",
        "max_mean_reward_degradation": 0.0,
        "max_exact_loss": 0,
        "max_nonzero_loss": 0,
        "deltas": deltas,
        "passed": (
            deltas["mean_reward"] >= 0.0
            and deltas["exact"] >= 0
            and deltas["nonzero"] >= 0
        ),
    }


def prose_summary(label, mean):
    return {
        "results_path": f"/safe/{label}.json",
        "results_sha256": digest(label),
        "item_count": 16,
        "scored_token_count": 10926,
        "mean_token_logprob": mean,
    }


def prose_gate(baseline, candidate, interval=None):
    delta = candidate["mean_token_logprob"] - baseline["mean_token_logprob"]
    if interval is None:
        interval = [delta, delta]
    return {
        "metric": "mean_token_logprob",
        "higher_is_better": True,
        "baseline": baseline["mean_token_logprob"],
        "final": candidate["mean_token_logprob"],
        "delta": delta,
        "max_degradation": 0.0,
        "paired_document_bootstrap_95_ci": interval,
        "bootstrap": {
            "unit": "normalized_source_url",
            "document_count": 16,
            "samples": 20000,
            "seed": 20260714,
            "percentiles": [0.025, 0.975],
        },
        "passed": interval[0] >= 0.0,
    }


def identity_audit(seed):
    weight_hash = digest("base-weights")
    identity = {
        "schema": "eggroll-es-weight-state-sha256-v2",
        "sha256": weight_hash,
        "parameter_count": 100,
        "total_bytes": 1000,
    }
    check = {
        "schema": "eggroll-es-exact-reference-check-v2",
        "passed": True,
        "reference": identity,
        "current": identity,
    }
    probe = {
        "schema": "eggroll-es-train-only-identity-probe-v2",
        "domain_output_sha256": digest(f"domain-probe-{seed}"),
        "anchor_output_sha256": digest(f"anchor-probe-{seed}"),
        "domain_requests": 64,
        "anchor_requests": 16,
    }
    return {
        "schema": "eggroll-es-alpha-zero-identity-audit-v2",
        "iteration": 0,
        "status": "passed",
        "training_signal": "train_batch_and_train_only_anchor_only",
        "reference_states": [[identity] for _ in range(4)],
        "pre_probe": probe,
        "post_probe": copy.deepcopy(probe),
        "post_reference_checks": [[check] for _ in range(4)],
        "passed": True,
    }


def snapshot(seed, targets, schema_version=2):
    implementation = {
        key: digest(key)
        for key in (
            "driver", "anchor_trainer", "base_trainer", "projection",
            "upstream_trainer", "upstream_worker", "corrected_driver",
            "exact_worker",
        )
    }
    value = {
        "schema": f"eggroll-es-anchor-line-search-snapshot-v{schema_version}",
        "train": {
            "rows": 794,
            "arrow_files": [{
                "path": "/safe/s6/train.arrow",
                "sha256": digest("train-arrow"),
            }],
        },
        "evaluations": {
            "validation": {
                "rows": 41,
                "arrow_files": [{
                    "path": "/safe/s6/validation.arrow",
                    "sha256": digest("validation-arrow"),
                }],
            },
            "ood_qa": {
                "rows": 24,
                "arrow_files": [{
                    "path": "/safe/s6/ood_qa.arrow",
                    "sha256": digest("ood-arrow"),
                }],
            },
        },
        "anchor": {
            "path": "/safe/anchor.jsonl",
            "sha256": digest("anchor"),
            "rows": 128,
            "report": {
                "path": "/safe/anchor.report.json",
                "sha256": digest("anchor-report"),
                "schema": "general-prose-anchor-build-v1",
                "protected_artifact_count": 4,
            },
        },
        "fixed_train_batch": {
            "rows": 64,
            "sha256": digest(f"fixed-batch-{seed}"),
        },
        "implementation": implementation,
        "recipe": {
            "model_name": "/safe/models/Qwen3.6-35B-A3B",
            "checkpoint": None,
            "sigma": 0.0003,
            "population_size": 8,
            "batch_size": 64,
            "mini_batch_size": 64,
            "max_tokens": 32,
            "seed": seed,
            "min_anchor_cosine": 0.25,
            "anchor_items_per_step": 16,
            "target_alphas": list(targets),
        },
    }
    if schema_version in (3, 4):
        value["implementation"].update({
            key: digest(key)
            for key in (
                "distributed_driver_v3",
                "distributed_trainer_v3",
                "distributed_worker_v3",
            )
        })
        value["distributed_update_v3"] = {
            "engine_count": 4,
            "tp_per_engine": 1,
            "seed_sharding": "strided_by_inter_engine_rank",
            "collective_dtype": "torch.float32",
            "two_phase_commit": True,
            "final_hash_consensus_required": True,
            "reference_recapture_policy": "once_before_next_population_only",
            "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
            "direct_alpha_confirmation_required": True,
        }
    if schema_version == 4:
        value["train"]["arrow_files"][0]["sha256"] = (
            "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"
        )
        value["evaluations"]["validation"]["arrow_files"][0]["sha256"] = (
            "19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db"
        )
        value["evaluations"]["ood_qa"]["arrow_files"][0]["sha256"] = (
            "b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced"
        )
        value["anchor"]["sha256"] = (
            "a693e23c48e558e9b72c30b0ae31f0b3e580a665371846978ad4d3eca7ef5f7d"
        )
        value["anchor"]["report"]["sha256"] = (
            "913ff2cb786ac50ffe86770291b6173a14220afce3682dfea67359c45cf6e9f5"
        )
        value["implementation"].update({
            key: digest(key)
            for key in (
                "distributed_driver_v4",
                "distributed_trainer_v4",
                "distributed_worker_v4",
            )
        })
        value["frozen_layer_plan_v4"] = {
            "path": "/safe/front_back_dense.json",
            "file_sha256": V4_PLAN_FILE_SHA,
            "plan_sha256": V4_PLAN_SHA,
            "model_config_path": "/safe/qwen-config.json",
            "model_config_sha256": V4_MODEL_CONFIG_SHA,
        }
        value["dense_gold_reward_v4"] = {
            "config": copy.deepcopy(V4_DENSE_REWARD_CONFIG),
            "reward_config_sha256": V4_DENSE_REWARD_SHA,
        }
        value["distributed_update_v4"] = {
            "engine_count": 4,
            "tp_per_engine": 1,
            "seed_sharding": "strided_by_inter_engine_rank",
            "collective_dtype": "torch.float32",
            "two_phase_commit": True,
            "final_hash_consensus_required": True,
            "reference_recapture_policy": "once_before_next_population_only",
            "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
            "direct_alpha_confirmation_required": True,
            "layer_plan_sha256": V4_PLAN_SHA,
            "dense_reward_sha256": V4_DENSE_REWARD_SHA,
        }
    return value


def attach_v3_provenance(journal):
    coefficient_plan = journal["coefficient_plan"]
    population_size = journal["trainer_configuration"]["population_size"]
    seeds = list(journal["seeds"])
    coefficients = [
        (index - population_size / 2.0) / population_size
        for index in range(population_size)
    ]
    coefficient_sha = canonical_sha256({
        "seeds": seeds,
        "coefficients": coefficients,
    })
    coefficient_plan["coefficient_sha256"] = coefficient_sha
    coefficient_plan["coefficients"] = coefficients
    for state in journal["states"]:
        state["coefficient_sha256"] = coefficient_sha
    reference_identity = copy.deepcopy(
        coefficient_plan["identity_audit"]["reference_states"][0][0]
    )
    reference_generation = 1
    plan_id = canonical_sha256({
        "schema": "eggroll-es-distributed-plan-id-v3",
        "iteration": 0,
        "coefficient_sha256": coefficient_sha,
        "reference_generation": reference_generation,
        "reference_sha256": reference_identity["sha256"],
    })
    coefficient_plan["distributed_update_v3"] = {
        "schema": "eggroll-es-distributed-seed-plan-v3",
        "plan_id": plan_id,
        "engine_count": 4,
        "tp_per_engine": 1,
        "reference_generation": reference_generation,
        "reference_identity": reference_identity,
        "seed_sharding": "strided_by_inter_engine_rank",
        "collective_dtype": "torch.float32",
        "reference_recapture_policy": "once_before_next_population_only",
    }
    applications = []
    previous_alpha = 0.0
    expected_base_sha = reference_identity["sha256"]
    for sequence, target_alpha in enumerate(journal["targets"][1:], 1):
        manifest = {
            "schema": "eggroll-es-distributed-update-manifest-v3",
            "coefficient_sha256": coefficient_sha,
            "population_size": population_size,
            "world_size": 4,
            "reference_generation": reference_generation,
            "plan_id": plan_id,
            "update_sequence": sequence,
            "previous_alpha": previous_alpha,
            "target_alpha": target_alpha,
            "expected_base_sha256": expected_base_sha,
        }
        manifest_sha = canonical_sha256(manifest)
        final_identity = {
            "schema": "eggroll-es-weight-state-sha256-v2",
            "sha256": digest(
                f"v3-final-{journal['trainer_configuration']['global_seed']}-{sequence}"
            ),
            "parameter_count": reference_identity["parameter_count"],
            "total_bytes": reference_identity["total_bytes"],
        }
        prepared = []
        executed = []
        commits = []
        post_states = []
        for rank in range(4):
            indices = list(range(rank, population_size, 4))
            prepared.append({
                "schema": "eggroll-es-distributed-update-prepared-v3",
                "prepared": True,
                "manifest_sha256": manifest_sha,
                "rank": rank,
                "world_size": 4,
                "shard_indices": indices,
                "shard_seeds": [seeds[index] for index in indices],
                "shard_pair_sha256": canonical_sha256({
                    "seeds": [seeds[index] for index in indices],
                    "coefficients": [
                        coefficients[index] for index in indices
                    ],
                }),
                "base_sha256": expected_base_sha,
                "reference_generation": reference_generation,
                "update_sequence": sequence,
                "allocation_preflight": {
                    "schema": "eggroll-es-local-allocation-preflight-v3",
                    "passed": True,
                    "parameter_count": 100,
                    "largest_parameter_name": "model.weight",
                    "largest_parameter_shape": [100],
                    "parameter_dtype": "torch.bfloat16",
                    "accumulator_dtype": "torch.float32",
                    "simulated_peak_temporary_bytes": 600,
                    "scratch_freed_before_collectives": True,
                    "collectives_created": False,
                    "rng_consumed": False,
                    "weights_changed": False,
                },
            })
            executed.append({
                "schema": "eggroll-es-distributed-update-executed-v3",
                "executed": True,
                "manifest_sha256": manifest_sha,
                "rank": rank,
                "world_size": 4,
                "parameter_count": 100,
                "reduced_element_count": 1000,
                "collective_dtype": "torch.float32",
                "final_identity": copy.deepcopy(final_identity),
            })
            commits.append({
                "schema": "eggroll-es-distributed-update-committed-v3",
                "committed": True,
                "manifest_sha256": manifest_sha,
                "rank": rank,
                "final_sha256": final_identity["sha256"],
                "reference_generation": reference_generation,
                "reference_fresh_for_population": False,
                "update_sequence": sequence,
                "accepted_alpha": target_alpha,
            })
            post_states.append({
                "schema": "eggroll-es-distributed-worker-state-v3",
                "communicator": {
                    "rank": rank,
                    "world_size": 4,
                    "tp_world_size": 1,
                    "available": True,
                    "disabled": False,
                },
                "reference_generation": reference_generation,
                "reference_fresh_for_population": False,
                "reference_identity": copy.deepcopy(reference_identity),
                "current_identity": copy.deepcopy(final_identity),
                "update_session": plan_id,
                "update_sequence": sequence,
                "accepted_alpha": target_alpha,
                "pending": False,
            })
        applications.append({
            "schema": "eggroll-es-distributed-alpha-application-v3",
            "target_alpha": target_alpha,
            "alpha_increment": target_alpha - previous_alpha,
            "update_sequence": sequence,
            "manifest_sha256": manifest_sha,
            "coefficient_sha256": coefficient_sha,
            "prepared_shards": prepared,
            "executed_collectives": executed,
            "commits": commits,
            "post_commit_states": post_states,
            "final_identity": final_identity,
            "reference_recaptured": False,
            "reference_fresh_for_population": False,
            "bf16_alpha_semantics": "path_dependent_monotonic_increment",
            "direct_alpha_confirmation_required": True,
        })
        expected_base_sha = final_identity["sha256"]
        previous_alpha = target_alpha
    coefficient_plan["applications"] = applications


def v4_bindings():
    return {
        "layer_plan_file_sha256": V4_PLAN_FILE_SHA,
        "layer_plan_sha256": V4_PLAN_SHA,
        "checkpoint_to_runtime_mapping_sha256": V4_MAPPING_SHA,
        "source_unit_count": 70,
        "runtime_selected_name_sha256": V4_RUNTIME_NAMES_SHA,
        "selected_parameter_manifest_sha256": digest("selected-manifest"),
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "unselected_origin_sha256": digest("unselected-origin"),
        "dense_reward_sha256": V4_DENSE_REWARD_SHA,
    }


def v4_partition(label, partition, bindings):
    selected = partition == "selected"
    value = {
        "schema": "eggroll-es-parameter-partition-sha256-v4",
        "partition": partition,
        "sha256": (
            digest(label)
            if selected else bindings["unselected_origin_sha256"]
        ),
        "parameter_count": 46 if selected else 100,
        "total_elements": 285_999_104 if selected else 1_000_000,
        "total_bytes": 571_998_208 if selected else 2_000_000,
        "layer_plan_file_sha256": bindings["layer_plan_file_sha256"],
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "checkpoint_to_runtime_mapping_sha256": (
            bindings["checkpoint_to_runtime_mapping_sha256"]
        ),
        "runtime_selected_name_sha256": (
            bindings["runtime_selected_name_sha256"]
        ),
        "selected_parameter_manifest_sha256": (
            bindings["selected_parameter_manifest_sha256"]
        ),
        "dense_reward_sha256": bindings["dense_reward_sha256"],
    }
    return value


def v4_weight_identity(label, bindings):
    selected = v4_partition(f"{label}-selected", "selected", bindings)
    unselected = v4_partition(f"{label}-unselected", "unselected", bindings)
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
    return {
        "schema": "eggroll-es-partitioned-weight-state-v4",
        "sha256": canonical_sha256(payload),
        **{key: payload[key] for key in (
            "layer_plan_file_sha256", "layer_plan_sha256",
            "checkpoint_to_runtime_mapping_sha256", "source_unit_count",
            "runtime_selected_name_sha256",
            "selected_parameter_manifest_sha256", "dense_reward_sha256",
            "selected", "unselected",
        )},
    }


def v4_identity_audit(seed, bindings, reference_identity):
    probe = {
        "schema": "eggroll-es-train-only-identity-probe-v4",
        "dense_gold_output_sha256": digest(f"dense-probe-{seed}"),
        "anchor_output_sha256": digest(f"anchor-probe-{seed}"),
        "domain_requests": 64,
        "anchor_requests": 16,
        "reward_config_sha256": bindings["dense_reward_sha256"],
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "dispatch": "strided_engine_shards_separate_calls",
    }
    states = []
    checks = []
    for _rank in range(4):
        states.append([{
            "schema": "eggroll-es-selected-exact-reference-state-v4",
            "reference_generation": 1,
            "fresh_for_population": True,
            "identity": copy.deepcopy(reference_identity),
            **bindings,
        }])
        checks.append([{
            "schema": "eggroll-es-selected-exact-reference-check-v4",
            "passed": True,
            "reference_generation": 1,
            "reference": copy.deepcopy(reference_identity["selected"]),
            "current": copy.deepcopy(reference_identity["selected"]),
            "unselected_audit": "deferred_to_population_completion_v4",
            **bindings,
        }])
    return {
        "schema": "eggroll-es-alpha-zero-identity-audit-v2",
        "iteration": 0,
        "status": "passed",
        "training_signal": "train_batch_and_train_only_anchor_only",
        "reference_states": states,
        "pre_probe": probe,
        "post_probe": copy.deepcopy(probe),
        "post_reference_checks": checks,
        "passed": True,
    }


def attach_v4_provenance(journal):
    coefficient_plan = journal["coefficient_plan"]
    population_size = journal["trainer_configuration"]["population_size"]
    seeds = list(journal["seeds"])
    coefficients = [
        (index - population_size / 2.0) / population_size
        for index in range(population_size)
    ]
    coefficient_sha = canonical_sha256({
        "seeds": seeds,
        "coefficients": coefficients,
    })
    coefficient_plan["coefficient_sha256"] = coefficient_sha
    coefficient_plan["coefficients"] = coefficients
    for state in journal["states"]:
        state["coefficient_sha256"] = coefficient_sha
    bindings = v4_bindings()
    reference = v4_weight_identity("v4-reference", bindings)
    coefficient_plan["identity_audit"] = v4_identity_audit(
        journal["trainer_configuration"]["global_seed"], bindings, reference,
    )
    coefficient_plan["frozen_layer_plan_v4"] = {
        **copy.deepcopy(journal["snapshot"]["frozen_layer_plan_v4"]),
        "runtime_mapping": copy.deepcopy(bindings),
        "runtime_mapping_sha256": canonical_sha256(bindings),
    }
    coefficient_plan["dense_gold_reward_v4"] = copy.deepcopy(
        journal["snapshot"]["dense_gold_reward_v4"],
    )
    boundary = {
        "schema": "eggroll-es-population-boundary-audit-v4",
        "iteration": 0,
        "phase": "after_complete_population_exact_restore_before_plan",
        "engine_count": 4,
        "reference_generation": 1,
        "reference_identity": copy.deepcopy(reference),
        "current_identity": copy.deepcopy(reference),
        "unselected_origin_sha256": bindings["unselected_origin_sha256"],
        "runtime_mapping": copy.deepcopy(bindings),
        "worker_reports": [{
            "schema": "eggroll-es-post-population-audit-v4",
            "passed": True,
            "rank": rank,
            "world_size": 4,
            "reference_generation": 1,
            "reference_sha256": reference["sha256"],
            "current_identity": copy.deepcopy(reference),
            **bindings,
        } for rank in range(4)],
        "passed": True,
    }
    boundary["audit_sha256"] = canonical_sha256(boundary)
    coefficient_plan["population_boundary_audit_v4"] = boundary
    plan_id = canonical_sha256({
        "schema": "eggroll-es-distributed-plan-id-v4",
        "iteration": 0,
        "coefficient_sha256": coefficient_sha,
        "reference_generation": 1,
        "reference_sha256": reference["sha256"],
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "reward_config_sha256": bindings["dense_reward_sha256"],
        "runtime_mapping_sha256": canonical_sha256(bindings),
        "population_boundary_audit_sha256": boundary["audit_sha256"],
    })
    coefficient_plan["distributed_update_v4"] = {
        "schema": "eggroll-es-distributed-seed-plan-v4",
        "plan_id": plan_id,
        "engine_count": 4,
        "tp_per_engine": 1,
        "reference_generation": 1,
        "reference_identity": copy.deepcopy(reference),
        "seed_sharding": "strided_by_inter_engine_rank",
        "collective_dtype": "torch.float32",
        "reference_recapture_policy": "once_before_next_population_only",
        "layer_plan_sha256": bindings["layer_plan_sha256"],
        "dense_reward_sha256": bindings["dense_reward_sha256"],
        "runtime_mapping": copy.deepcopy(bindings),
        "runtime_mapping_sha256": canonical_sha256(bindings),
        "population_boundary_audit_sha256": boundary["audit_sha256"],
    }
    applications = []
    previous_alpha = 0.0
    previous_identity = reference
    for sequence, target_alpha in enumerate(journal["targets"][1:], 1):
        manifest = {
            "schema": "eggroll-es-layer-restricted-update-manifest-v4",
            "coefficient_sha256": coefficient_sha,
            "population_size": population_size,
            "world_size": 4,
            "reference_generation": 1,
            "plan_id": plan_id,
            "update_sequence": sequence,
            "previous_alpha": previous_alpha,
            "target_alpha": target_alpha,
            "expected_base_sha256": previous_identity["sha256"],
            **bindings,
        }
        manifest_sha = canonical_sha256(manifest)
        final_identity = v4_weight_identity(
            f"v4-final-{journal['trainer_configuration']['global_seed']}-{sequence}",
            bindings,
        )
        prepared = []
        executed = []
        commits = []
        post_states = []
        for rank in range(4):
            indices = list(range(rank, population_size, 4))
            prepared.append({
                "schema": "eggroll-es-layer-restricted-update-prepared-v4",
                "prepared": True,
                "manifest_sha256": manifest_sha,
                "rank": rank,
                "world_size": 4,
                "shard_indices": indices,
                "shard_seeds": [seeds[index] for index in indices],
                "shard_pair_sha256": canonical_sha256({
                    "seeds": [seeds[index] for index in indices],
                    "coefficients": [
                        coefficients[index] for index in indices
                    ],
                }),
                "base_sha256": previous_identity["sha256"],
                "reference_generation": 1,
                "update_sequence": sequence,
                "allocation_preflight": {
                    "schema": "eggroll-es-selected-allocation-preflight-v4",
                    "passed": True,
                    "parameter_count": 46,
                    "element_count": 285_999_104,
                    "largest_parameter_name": "model.selected.weight",
                    "largest_parameter_shape": [1024, 1024],
                    "parameter_dtype": "torch.bfloat16",
                    "accumulator_dtype": "torch.float32",
                    "simulated_peak_temporary_bytes": 6_291_456,
                    "scratch_freed_before_collectives": True,
                    "collectives_created": False,
                    "rng_consumed": False,
                    "weights_changed": False,
                    **bindings,
                },
                **bindings,
            })
            executed.append({
                "schema": "eggroll-es-layer-restricted-update-executed-v4",
                "executed": True,
                "manifest_sha256": manifest_sha,
                "rank": rank,
                "world_size": 4,
                "parameter_count": 46,
                "reduced_element_count": 285_999_104,
                "collective_dtype": "torch.float32",
                "final_identity": copy.deepcopy(final_identity),
                **bindings,
            })
            commits.append({
                "schema": "eggroll-es-layer-restricted-update-committed-v4",
                "committed": True,
                "manifest_sha256": manifest_sha,
                "rank": rank,
                "final_sha256": final_identity["sha256"],
                "reference_generation": 1,
                "reference_fresh_for_population": False,
                "update_sequence": sequence,
                "accepted_alpha": target_alpha,
                **bindings,
            })
            post_states.append({
                "schema": "eggroll-es-layer-restricted-worker-state-v4",
                "communicator": {
                    "rank": rank,
                    "world_size": 4,
                    "tp_world_size": 1,
                    "available": True,
                    "disabled": False,
                },
                "reference_generation": 1,
                "reference_fresh_for_population": False,
                "reference_identity": copy.deepcopy(reference),
                "current_identity": copy.deepcopy(final_identity),
                "update_session": plan_id,
                "update_sequence": sequence,
                "accepted_alpha": target_alpha,
                "pending": False,
                **bindings,
            })
        applications.append({
            "schema": "eggroll-es-distributed-alpha-application-v4",
            "target_alpha": target_alpha,
            "alpha_increment": target_alpha - previous_alpha,
            "update_sequence": sequence,
            "manifest_sha256": manifest_sha,
            "manifest": manifest,
            "coefficient_sha256": coefficient_sha,
            "layer_plan_sha256": bindings["layer_plan_sha256"],
            "dense_reward_sha256": bindings["dense_reward_sha256"],
            "runtime_mapping": copy.deepcopy(bindings),
            "prepared_shards": prepared,
            "executed_collectives": executed,
            "commits": commits,
            "post_commit_states": post_states,
            "final_identity": final_identity,
            "reference_recaptured": False,
            "reference_fresh_for_population": False,
            "bf16_alpha_semantics": "path_dependent_monotonic_increment",
            "direct_alpha_confirmation_required": True,
        })
        previous_alpha = target_alpha
        previous_identity = final_identity
    coefficient_plan["applications"] = applications


def make_journal(
    seed,
    validation_delta=0.01,
    ood_mean_delta=0.0,
    ood_exact_delta=0,
    ood_nonzero_delta=0,
    prose_delta=0.001,
    prose_interval=None,
    targets=(0.0, 0.0000125),
    schema_version=2,
):
    baseline_validation_mean = (
        0.08381010452961674 if schema_version == 4 else 0.08
    )
    baseline_ood_mean = (
        0.714128787878788 if schema_version == 4 else 0.7
    )
    baseline_prose_mean = (
        -1.2632580042542214 if schema_version == 4 else -1.26
    )
    baseline_validation = qa_summary(
        f"validation-base-{seed}", baseline_validation_mean, 2, 13, 41,
    )
    baseline_ood = qa_summary(
        f"ood-base-{seed}", baseline_ood_mean, 16, 23, 24,
    )
    baseline_prose = prose_summary(
        f"prose-base-{seed}", baseline_prose_mean,
    )
    coefficient_hash = digest(f"coefficients-{seed}")
    states = []
    previous = 0.0
    for index, target in enumerate(targets):
        fraction = 0.0 if target == 0.0 else target / targets[-1]
        validation = qa_summary(
            f"validation-{seed}-{index}",
            baseline_validation["mean_reward"] + validation_delta * fraction,
            2, 13, 41,
        )
        ood = qa_summary(
            f"ood-{seed}-{index}",
            baseline_ood["mean_reward"] + ood_mean_delta * fraction,
            16 + (ood_exact_delta if index else 0),
            23 + (ood_nonzero_delta if index else 0),
            24,
        )
        prose = prose_summary(
            f"prose-{seed}-{index}",
            baseline_prose["mean_token_logprob"] + prose_delta * fraction,
        )
        q_gate = qa_gate(baseline_ood, ood)
        interval = (
            [0.0, 0.0] if index == 0 else prose_interval
        )
        p_gate = prose_gate(baseline_prose, prose, interval)
        states.append({
            "state_index": index,
            "target_alpha": target,
            "alpha_increment": target - previous,
            "eval_iteration": index,
            "coefficient_sha256": coefficient_hash,
            "qa": {"validation": validation, "ood_qa": ood},
            "ood_qa_gate": q_gate,
            "ood_prose": prose,
            "ood_prose_gate": p_gate,
            "strict_guards_passed": q_gate["passed"] and p_gate["passed"],
        })
        previous = target
    journal = {
        "schema": f"eggroll-es-anchor-alpha-line-search-v{schema_version}",
        "status": "complete",
        "policy": {
            "alpha_order": "zero_then_strictly_increasing",
            "branching": False,
            "resume": False,
            "rollback": False,
            "selection_during_execution": False,
            "ood_qa_max_degradation": 0.0,
            "ood_prose_max_degradation": 0.0,
        },
        "targets": list(targets),
        "trainer_configuration": {
            "model_name": "/safe/models/Qwen3.6-35B-A3B",
            "sigma": 0.0003,
            "population_size": 8,
            "batch_size": 64,
            "mini_batch_size": 64,
            "max_tokens": 32,
            "global_seed": seed,
            "min_anchor_cosine": 0.25,
            "anchor_items_per_step": 16,
        },
        "snapshot": snapshot(seed, targets, schema_version),
        "coefficient_plan": {
            "coefficient_sha256": coefficient_hash,
            "journal_path": f"/safe/plan-{seed}.json",
            "projection": {"decision": "project_to_anchor_cone"},
            "seed_count": 8,
            "identity_audit": identity_audit(seed),
        },
        "in_progress": None,
        "states": states,
        "seeds": [seed * 100 + index for index in range(8)],
    }
    if schema_version in (3, 4):
        journal["policy"].update({
            "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
            "direct_alpha_confirmation_required": True,
        })
    if schema_version == 3:
        attach_v3_provenance(journal)
    elif schema_version == 4:
        journal["policy"].update({
            "frozen_layer_plan_required": True,
            "dense_gold_reward_required": True,
        })
        attach_v4_provenance(journal)
    journal["content_sha256_before_self_field"] = canonical_sha256(journal)
    return journal


def reseal(journal):
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = canonical_sha256(journal)
    return journal


def five_journals(deltas=None, **kwargs):
    if deltas is None:
        deltas = [0.010, 0.009, 0.008, 0.007, -0.001]
    return [
        make_journal(seed, validation_delta=delta, **kwargs)
        for seed, delta in zip((42, 43, 44, 45, 46), deltas)
    ]


def test_sorted_json_split_order_remains_valid_but_split_set_is_exact():
    journal = make_journal(42)
    journal["snapshot"]["evaluations"] = {
        key: journal["snapshot"]["evaluations"][key]
        for key in sorted(journal["snapshot"]["evaluations"])
    }
    for state in journal["states"]:
        state["qa"] = {key: state["qa"][key] for key in sorted(state["qa"])}
    reseal(journal)
    assert validate_journal(journal)["seed"] == 42

    journal["snapshot"]["evaluations"]["held-back"] = (
        journal["snapshot"]["evaluations"]["validation"]
    )
    reseal(journal)
    with pytest.raises(JournalValidationError, match="exactly validation,ood_qa"):
        validate_journal(journal)


def test_five_direct_confirmations_pass_all_predeclared_rules():
    assert "distributed_update_v3" not in validate_journal(
        make_journal(42)
    )
    report = aggregate_direct_confirmations(
        five_journals(), candidate_name="items16-cos025",
    )
    assert report["direct_confirmation"] is True
    assert report["path_dependent_pilot_states_counted"] is False
    assert report["aggregate_validation"]["positive_seed_count"] == 4
    assert report["aggregate_validation"]["median_delta"] > 0
    assert report["aggregate_validation"]["risk_adjusted_score"] > 0
    assert report["eligible"] is True
    assert report["selected"] == "items16-cos025"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mean", -0.001, "QA gate decision"),
        ("exact", -1, "QA gate decision"),
        ("nonzero", -1, "QA gate decision"),
    ],
)
def test_explicit_ood_qa_mean_exact_nonzero_failures_are_rejected(
    field, value, message,
):
    kwargs = {
        "ood_mean_delta": value if field == "mean" else 0.0,
        "ood_exact_delta": value if field == "exact" else 0,
        "ood_nonzero_delta": value if field == "nonzero" else 0,
    }
    journal = make_journal(42, **kwargs)
    # Simulate a lying producer; aggregation must recompute the decision.
    journal["states"][1]["ood_qa_gate"]["passed"] = True
    journal["states"][1]["strict_guards_passed"] = True
    reseal(journal)
    with pytest.raises(JournalValidationError, match=message):
        validate_journal(journal)


def test_negative_prose_point_delta_is_rejected_even_with_nonnegative_bound():
    journal = make_journal(
        42, prose_delta=-0.001, prose_interval=[0.0, 0.001],
    )
    # The old producer's lower-bound-only Boolean can still say true.
    journal["states"][1]["strict_guards_passed"] = True
    reseal(journal)
    with pytest.raises(JournalValidationError, match="strict guard decision"):
        validate_journal(journal)


def test_negative_prose_lower_bound_makes_candidate_ineligible():
    journals = five_journals(prose_interval=[-0.0001, 0.002])
    report = aggregate_direct_confirmations(
        journals, candidate_name="prose-failure",
    )
    assert report["all_strict_ood_guards_passed"] is False
    assert report["eligible"] is False
    assert report["selected"] == "baseline"


def test_monotonic_pilot_cannot_masquerade_as_direct_confirmation():
    journals = [
        make_journal(seed, targets=(0.0, 0.00000625, 0.0000125))
        for seed in (42, 43, 44, 45, 46)
    ]
    with pytest.raises(JournalValidationError, match="pilot states"):
        aggregate_direct_confirmations(journals, candidate_name="not-direct")
    pilot = summarize_pilot(journals[0])
    assert pilot["classification"] == "exploratory_monotonic_pilot"
    assert pilot["direct_confirmation"] is False
    assert pilot["selection_allowed"] is False


@pytest.mark.parametrize("mutation", ["train", "eval", "recipe", "implementation"])
def test_mixed_family_provenance_is_rejected(mutation):
    journals = five_journals()
    snapshot_value = journals[-1]["snapshot"]
    if mutation == "train":
        snapshot_value["train"]["arrow_files"][0]["sha256"] = digest("other-train")
    elif mutation == "eval":
        snapshot_value["evaluations"]["validation"]["arrow_files"][0]["sha256"] = digest("other-eval")
    elif mutation == "recipe":
        snapshot_value["recipe"]["sigma"] = 0.0005
        journals[-1]["trainer_configuration"]["sigma"] = 0.0005
    else:
        snapshot_value["implementation"]["exact_worker"] = digest("other-worker")
    reseal(journals[-1])
    with pytest.raises(JournalValidationError, match="identity differs"):
        aggregate_direct_confirmations(journals, candidate_name="mixed")


@pytest.mark.parametrize("status,in_progress", [("failed", None), ("complete", {"phase": "eval"})])
def test_failed_or_incomplete_journal_is_rejected(status, in_progress):
    journal = make_journal(42)
    journal["status"] = status
    journal["in_progress"] = in_progress
    reseal(journal)
    with pytest.raises(JournalValidationError, match="incomplete|in-progress"):
        validate_journal(journal)


def test_failed_exact_identity_audit_is_rejected():
    journal = make_journal(42)
    journal["coefficient_plan"]["identity_audit"]["status"] = "failed"
    journal["coefficient_plan"]["identity_audit"]["passed"] = False
    reseal(journal)
    with pytest.raises(JournalValidationError, match="did not pass"):
        validate_journal(journal)


def test_heldout_split_or_path_is_rejected_without_opening_it():
    journal = make_journal(42)
    journal["snapshot"]["evaluations"]["heldout"] = {
        "rows": 1,
        "arrow_files": [{
            "path": "/does/not/exist/sealed.arrow",
            "sha256": digest("sealed"),
        }],
    }
    reseal(journal)
    with pytest.raises(JournalValidationError, match="heldout"):
        validate_journal(journal)


def test_content_hash_tampering_is_rejected():
    journal = make_journal(42)
    journal["states"][1]["qa"]["validation"]["mean_reward"] += 1.0
    with pytest.raises(JournalValidationError, match="content hash"):
        validate_journal(journal)


def test_five_seed_positive_fraction_and_risk_rules_fail_closed():
    too_few_positive = five_journals(
        deltas=[0.010, 0.009, 0.008, -0.001, -0.001],
    )
    report = aggregate_direct_confirmations(
        too_few_positive, candidate_name="three-of-five",
    )
    assert report["aggregate_validation"]["positive_seed_count"] == 3
    assert report["eligible"] is False

    high_variance = five_journals(
        deltas=[0.001, 0.001, 0.001, 0.001, -0.02],
    )
    report = aggregate_direct_confirmations(
        high_variance, candidate_name="high-variance",
    )
    assert report["aggregate_validation"]["positive_seed_count"] == 4
    assert report["aggregate_validation"]["risk_adjusted_score"] <= 0
    assert report["eligible"] is False


def test_corrected_v3_journals_are_supported_as_a_separate_family():
    journals = [
        make_journal(seed, schema_version=3)
        for seed in (42, 43, 44, 45, 46)
    ]
    report = aggregate_direct_confirmations(journals, candidate_name="v3")
    assert report["eligible"] is True
    validated = validate_journal(journals[0])
    assert validated["distributed_update_v3"] == {
        "plan_id": journals[0]["coefficient_plan"]["distributed_update_v3"][
            "plan_id"
        ],
        "reference_generation": 1,
        "application_count": 1,
        "final_identity_sha256": journals[0]["coefficient_plan"][
            "applications"
        ][0]["final_identity"]["sha256"],
    }


def test_v3_monotonic_application_chain_binds_each_prior_final_hash():
    journal = make_journal(
        42,
        schema_version=3,
        targets=(0.0, 0.00000625, 0.0000125),
    )
    validated = validate_journal(journal)
    assert validated["distributed_update_v3"]["application_count"] == 2
    applications = journal["coefficient_plan"]["applications"]
    assert applications[1]["prepared_shards"][0]["base_sha256"] == (
        applications[0]["final_identity"]["sha256"]
    )

    applications[1]["prepared_shards"][0]["base_sha256"] = (
        journal["coefficient_plan"]["distributed_update_v3"][
            "reference_identity"
        ]["sha256"]
    )
    reseal(journal)
    with pytest.raises(JournalValidationError, match="prepared metadata"):
        validate_journal(journal)


def test_merely_relabelled_v2_journal_cannot_masquerade_as_v3():
    journal = make_journal(42)
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v3"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v3"
    )
    journal["policy"].update({
        "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
        "direct_alpha_confirmation_required": True,
    })
    reseal(journal)
    with pytest.raises(
        JournalValidationError, match="v3 implementation identity",
    ):
        validate_journal(journal)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("snapshot_worker", "implementation distributed_worker_v3"),
        ("snapshot_policy", "v3 snapshot distributed policy"),
        ("journal_policy", "v3 journal policy"),
        ("applications", "application count"),
        ("sequence", "sequence changed"),
        ("alpha", "alpha transition"),
        ("coefficient", "coefficient identity"),
        ("coefficient_values", "canonical coefficient values"),
        ("manifest", "manifest does not match"),
        ("prepared_ranks", "prepared reports ranks"),
        ("prepared_shard", "seed shard changed"),
        ("shard_pair", "seed/coefficient shard SHA-256"),
        ("preflight", "allocation preflight"),
        ("executed_dtype", "collective metadata"),
        ("executed_identity", "final identity differs"),
        ("commit_ranks", "commit reports ranks"),
        ("post_ranks", "post-commit states ranks"),
        ("post_identity", "current identity differs"),
        ("recapture", "recaptured"),
        ("direct_confirmation", "direct-confirmation policy"),
    ],
)
def test_v3_distributed_provenance_tampering_is_rejected(mutation, match):
    journal = make_journal(42, schema_version=3)
    snapshot_value = journal["snapshot"]
    policy = journal["policy"]
    plan = journal["coefficient_plan"]
    application = plan["applications"][0]
    if mutation == "snapshot_worker":
        snapshot_value["implementation"]["distributed_worker_v3"] = "bad"
    elif mutation == "snapshot_policy":
        snapshot_value["distributed_update_v3"]["two_phase_commit"] = False
    elif mutation == "journal_policy":
        policy["direct_alpha_confirmation_required"] = False
    elif mutation == "applications":
        plan["applications"] = []
    elif mutation == "sequence":
        application["update_sequence"] = 2
    elif mutation == "alpha":
        application["alpha_increment"] *= 2
    elif mutation == "coefficient":
        application["coefficient_sha256"] = digest("other-coefficient")
    elif mutation == "coefficient_values":
        plan["coefficients"][0] += 0.5
    elif mutation == "manifest":
        application["manifest_sha256"] = digest("other-manifest")
        for section in ("prepared_shards", "executed_collectives", "commits"):
            for report in application[section]:
                report["manifest_sha256"] = application["manifest_sha256"]
    elif mutation == "prepared_ranks":
        application["prepared_shards"][3]["rank"] = 2
    elif mutation == "prepared_shard":
        application["prepared_shards"][0]["shard_seeds"][0] += 1
    elif mutation == "shard_pair":
        application["prepared_shards"][0]["shard_pair_sha256"] = digest(
            "other-shard-pair"
        )
    elif mutation == "preflight":
        application["prepared_shards"][0]["allocation_preflight"][
            "collectives_created"
        ] = True
    elif mutation == "executed_dtype":
        application["executed_collectives"][0]["collective_dtype"] = (
            "torch.bfloat16"
        )
    elif mutation == "executed_identity":
        application["executed_collectives"][0]["final_identity"]["sha256"] = (
            digest("drifted-final")
        )
    elif mutation == "commit_ranks":
        application["commits"][3]["rank"] = 2
    elif mutation == "post_ranks":
        application["post_commit_states"][3]["communicator"]["rank"] = 2
    elif mutation == "post_identity":
        application["post_commit_states"][0]["current_identity"]["sha256"] = (
            digest("drifted-current")
        )
    elif mutation == "recapture":
        application["reference_recaptured"] = True
    else:
        application["direct_alpha_confirmation_required"] = False
    reseal(journal)
    with pytest.raises(JournalValidationError, match=match):
        validate_journal(journal)


def test_v4_journals_are_supported_as_a_separately_bound_family():
    journals = [
        make_journal(seed, schema_version=4)
        for seed in (42, 43, 44, 45, 46)
    ]
    report = aggregate_direct_confirmations(journals, candidate_name="v4-front-back")
    assert report["eligible"] is True
    validated = validate_journal(journals[0])
    provenance = validated["distributed_update_v4"]
    assert provenance["plan_id"] == journals[0]["coefficient_plan"][
        "distributed_update_v4"
    ]["plan_id"]
    assert provenance["layer_plan_sha256"] == V4_PLAN_SHA
    assert provenance["dense_reward_sha256"] == V4_DENSE_REWARD_SHA
    assert provenance["application_count"] == 1
    assert provenance["population_boundary_audit_sha256"] == journals[0][
        "coefficient_plan"
    ]["population_boundary_audit_v4"]["audit_sha256"]


def test_v4_accepts_only_both_predeclared_parameter_matched_runtime_plans():
    front = v4_bindings()
    assert _validate_v4_bindings(front, "front") == front
    middle = {
        **front,
        "layer_plan_file_sha256": V4_MIDDLE_PLAN_FILE_SHA,
        "layer_plan_sha256": V4_MIDDLE_PLAN_SHA,
        "checkpoint_to_runtime_mapping_sha256": V4_MIDDLE_MAPPING_SHA,
        "runtime_selected_name_sha256": V4_MIDDLE_RUNTIME_NAMES_SHA,
    }
    assert _validate_v4_bindings(middle, "middle") == middle
    middle["layer_plan_file_sha256"] = V4_PLAN_FILE_SHA
    with pytest.raises(JournalValidationError, match="file/plan identity pair"):
        _validate_v4_bindings(middle, "mismatched")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("train", "train artifact differs from frozen S6"),
        ("validation", "validation artifact differs from frozen S6"),
        ("eval_batch", "frozen recipe changed"),
        ("baseline", "alpha-zero validation did not reproduce frozen S6"),
    ],
)
def test_v4_rejects_s6_snapshot_recipe_or_baseline_drift(mutation, match):
    journal = make_journal(42, schema_version=4)
    if mutation == "train":
        journal["snapshot"]["train"]["arrow_files"][0]["sha256"] = digest(
            "different-train",
        )
    elif mutation == "validation":
        journal["snapshot"]["evaluations"]["validation"][
            "arrow_files"
        ][0]["sha256"] = digest("different-validation")
    elif mutation == "eval_batch":
        journal["snapshot"]["recipe"]["mini_batch_size"] = 8
        journal["trainer_configuration"]["mini_batch_size"] = 8
    else:
        journal["states"][0]["qa"]["validation"]["mean_reward"] += 0.001
    reseal(journal)
    with pytest.raises(JournalValidationError, match=match):
        validate_journal(journal)


def test_v4_monotonic_application_chain_binds_prior_partition_identity():
    journal = make_journal(
        42, schema_version=4,
        targets=(0.0, 0.00000625, 0.0000125),
    )
    validated = validate_journal(journal)
    assert validated["distributed_update_v4"]["application_count"] == 2
    applications = journal["coefficient_plan"]["applications"]
    assert applications[1]["manifest"]["expected_base_sha256"] == (
        applications[0]["final_identity"]["sha256"]
    )
    applications[1]["manifest"]["expected_base_sha256"] = journal[
        "coefficient_plan"
    ]["distributed_update_v4"]["reference_identity"]["sha256"]
    reseal(journal)
    with pytest.raises(JournalValidationError, match="manifest payload"):
        validate_journal(journal)


def test_merely_relabelled_v3_journal_cannot_masquerade_as_v4():
    journal = make_journal(42, schema_version=3)
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v4"
    journal["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v4"
    journal["policy"].update({
        "frozen_layer_plan_required": True,
        "dense_gold_reward_required": True,
    })
    reseal(journal)
    with pytest.raises(JournalValidationError, match="v4 implementation identity"):
        validate_journal(journal)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("snapshot_plan", "file/plan identity pair"),
        ("snapshot_reward", "dense reward semantics"),
        ("runtime_mapping", "checkpoint/runtime mapping"),
        ("identity_selected_size", "frozen BF16 size"),
        ("boundary_hash", "boundary audit hash"),
        ("boundary_worker", "partition payload"),
        ("plan_id", "plan ID"),
        ("manifest", "manifest payload"),
        ("prepared_shard", "seed/coefficient shard"),
        ("preflight", "selected-only and side-effect free"),
        ("executed_count", "collective metadata"),
        ("executed_identity", "partition payload"),
        ("commit_binding", "commit binding"),
        ("post_reference", "immutable origin"),
        ("recapture", "reference or direct-confirmation policy"),
    ],
)
def test_v4_plan_reward_runtime_update_and_audit_tampering_is_rejected(
    mutation, match,
):
    journal = make_journal(42, schema_version=4)
    snapshot_value = journal["snapshot"]
    plan = journal["coefficient_plan"]
    application = plan["applications"][0]
    if mutation == "snapshot_plan":
        snapshot_value["frozen_layer_plan_v4"]["file_sha256"] = digest(
            "other-plan-file"
        )
    elif mutation == "snapshot_reward":
        snapshot_value["dense_gold_reward_v4"]["config"]["objective"] = (
            "generated_answer_reward"
        )
    elif mutation == "runtime_mapping":
        plan["frozen_layer_plan_v4"]["runtime_mapping"][
            "checkpoint_to_runtime_mapping_sha256"
        ] = digest("other-runtime-mapping")
    elif mutation == "identity_selected_size":
        plan["identity_audit"]["reference_states"][0][0]["identity"][
            "selected"
        ]["total_bytes"] -= 2
    elif mutation == "boundary_hash":
        plan["population_boundary_audit_v4"]["audit_sha256"] = digest(
            "other-boundary-audit"
        )
    elif mutation == "boundary_worker":
        plan["population_boundary_audit_v4"]["worker_reports"][0][
            "current_identity"
        ]["selected"]["sha256"] = digest("drifted-boundary-selected")
    elif mutation == "plan_id":
        plan["distributed_update_v4"]["plan_id"] = digest("other-plan-id")
    elif mutation == "manifest":
        application["manifest"]["target_alpha"] *= 2
    elif mutation == "prepared_shard":
        application["prepared_shards"][0]["shard_pair_sha256"] = digest(
            "other-shard-pair"
        )
    elif mutation == "preflight":
        application["prepared_shards"][0]["allocation_preflight"][
            "collectives_created"
        ] = True
    elif mutation == "executed_count":
        application["executed_collectives"][0]["reduced_element_count"] -= 1
    elif mutation == "executed_identity":
        application["executed_collectives"][0]["final_identity"]["selected"][
            "sha256"
        ] = digest("drifted-executed-selected")
    elif mutation == "commit_binding":
        application["commits"][0]["dense_reward_sha256"] = digest(
            "other-reward"
        )
    elif mutation == "post_reference":
        application["post_commit_states"][0]["reference_identity"][
            "unselected"
        ]["sha256"] = digest("drifted-unselected")
    else:
        application["reference_recaptured"] = True
    reseal(journal)
    with pytest.raises(JournalValidationError, match=match):
        validate_journal(journal)
