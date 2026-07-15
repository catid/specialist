#!/usr/bin/env python3
"""One equal-conflict-unit, nonzero EGGROLL-ES update for V37 fold 3."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import build_train_shadow_folds_v37a as shadow
import eggroll_es_worker_v38a as worker_v38a
import run_sft_train_only_control_v36a as hashing
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v13 as anchor_v13
from qa_quality import qa_pair_from_record


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v38a.EqualUnitUpdateWorkerExtensionV38A"
)
REQUIRED_ENGINE_COUNT = 4
POPULATION_SIZE = 32
SIGMA = 0.0003
ALPHA = 0.00015
SEEDS = list(anchor_v13.PERTURBATION_SEEDS_V13)
anchor_v4 = anchor_v13.anchor_v4
anchor_v10 = anchor_v13.anchor_v10
canonical_sha256 = anchor_v13.canonical_sha256
coefficient_sha256 = anchor_v13.coefficient_sha256


def create_placement_groups_v38a(placement_group_fn, num_engines: int):
    """Create four driver-scoped GPU reservations; detached lifetime is forbidden."""
    if int(num_engines) != REQUIRED_ENGINE_COUNT:
        raise ValueError("v38a requires exactly four placement groups")
    return [
        placement_group_fn(
            [{"GPU": 1, "CPU": 0}],
            strategy="PACK",
        )
        for _index in range(REQUIRED_ENGINE_COUNT)
    ]


def load_equal_unit_train_bundle(
    dataset_path: Path,
    dataset_sha256: str,
    manifest_path: Path,
    manifest_file_sha256: str,
) -> dict:
    dataset_path = Path(dataset_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if hashing.file_sha256(dataset_path) != dataset_sha256:
        raise RuntimeError("v38a training dataset identity changed")
    if hashing.file_sha256(manifest_path) != manifest_file_sha256:
        raise RuntimeError("v38a split-manifest identity changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("content_sha256_before_self_field")
        != shadow.EXPECTED_MANIFEST_CONTENT_SHA256
        or manifest.get("selection_firewall", {}).get("confirmatory_fold") != 3
    ):
        raise RuntimeError("v38a confirmatory split contract changed")
    rows = [
        json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 448:
        raise RuntimeError("v38a fold-3 training row count changed")
    row_to_unit = {}
    for unit in manifest["content_free_unit_commitments"]:
        if unit["fold"] == 3:
            continue
        for row_sha in unit["row_sha256"]:
            if row_sha in row_to_unit:
                raise RuntimeError("v38a row belongs to multiple training units")
            row_to_unit[row_sha] = {
                "unit_identity_sha256": unit["unit_identity_sha256"],
                "unit_rows": unit["row_count"],
                "stratum": unit["dominant_stratum"],
            }
    row_hashes = [shadow.row_sha256(row) for row in rows]
    if set(row_hashes) != set(row_to_unit) or len(row_to_unit) != 448:
        raise RuntimeError("v38a manifest row-to-unit commitment changed")
    unit_ids = {value["unit_identity_sha256"] for value in row_to_unit.values()}
    if len(unit_ids) != 208:
        raise RuntimeError("v38a training conflict-unit count changed")
    questions = []
    answers = []
    weights = []
    for row, row_sha in zip(rows, row_hashes):
        pair = qa_pair_from_record(row)
        if pair is None:
            raise RuntimeError("v38a training row is not QA")
        questions.append(pair[0])
        answers.append(pair[1])
        unit = row_to_unit[row_sha]
        weights.append(1.0 / (208 * unit["unit_rows"]))
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("v38a equal-unit weights do not sum to one")
    result = {
        "schema": "eggroll-es-equal-unit-train-bundle-v38a",
        "dataset": {
            "path": str(dataset_path), "file_sha256": dataset_sha256,
            "rows": 448, "ordered_row_sha256": canonical_sha256(row_hashes),
        },
        "split_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha256,
            "content_sha256": manifest["content_sha256_before_self_field"],
        },
        "questions": questions,
        "answers": answers,
        "weights": weights,
        "row_sha256": row_hashes,
        "conflict_units": 208,
        "weight_identity_sha256": canonical_sha256([
            {
                "row_sha256": row_sha,
                **row_to_unit[row_sha],
            } for row_sha in row_hashes
        ]),
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _score_equal_unit(bundle: dict, dense_items: list, outputs: list) -> dict:
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = [item["mean_answer_token_logprob"] for item in dense["examples"]]
    weights = bundle["weights"]
    return {
        "equal_unit_mean": math.fsum(
            weight * reward for weight, reward in zip(weights, rewards)
        ),
        "unweighted_row_mean": math.fsum(rewards) / len(rewards),
        "dense_result_sha256": canonical_sha256(dense),
        "scored_answer_tokens": dense["answer_token_count"],
    }


class EqualUnitUpdateMixinV38A:
    def configure_equal_unit_v38a(self, train_bundle, *, frozen_layer_plan):
        train_bundle = copy.deepcopy(train_bundle)
        anchor_v13.validate_frozen_layer_plan_bundle_v13(frozen_layer_plan)
        if (
            len(self.engines) != 4
            or int(self.n_vllm_engines) != 4
            or int(self.n_gpu_per_vllm_engine) != 1
            or self.population_size != 32
            or not math.isclose(float(self.sigma), SIGMA, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RuntimeError("v38a exact four-engine population recipe changed")
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
            raise RuntimeError("v38a workers captured different references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4", (4, "exact_reference"),
        )
        summary = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(summary, fresh=True)
        self._v38a_train_bundle = train_bundle
        identities = self._rpc_all_engines_v4("runtime_identity_v38a", ())
        return {
            "layer_plan_install": copy.deepcopy(install),
            "reference_identity": copy.deepcopy(summary["reference_identity"]),
            "train_bundle_content_sha256": train_bundle[
                "content_sha256_before_self_field"
            ],
            "worker_identities": identities,
        }

    def _prepared_v38a(self):
        bundle = self._v38a_train_bundle
        prompts = [base.specialist_template(question) for question in bundle["questions"]]
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, prompts, bundle["answers"],
        )
        return bundle, dense_items, [
            {"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items
        ]

    def _base_probe_v38a(self, bundle, dense_items, prompts):
        outputs = anchor_v13.anchor_v11.anchor_v1.dispatch_eval_batch(
            self.engines, prompts, self._dense_sampling_params_v4(0), self._resolve,
        )
        return _score_equal_unit(bundle, dense_items, outputs)["dense_result_sha256"]

    def estimate_apply_and_snapshot_v38a(self, snapshot_path, target_alpha=ALPHA):
        if float(target_alpha) != ALPHA:
            raise RuntimeError("v38a fixed alpha changed")
        bundle, dense_items, prompts = self._prepared_v38a()
        pre_probe = self._base_probe_v38a(bundle, dense_items, prompts)
        signed = {"plus": [], "minus": []}
        unweighted = {"plus": [], "minus": []}
        dense_hashes = {"plus": [], "minus": []}
        for start in range(0, len(SEEDS), 4):
            wave = anchor_v10.validate_full_engine_wave_v10(SEEDS[start:start + 4], 4)
            for sign, negate in (("plus", False), ("minus", True)):
                batches = None
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights", args=(int(seed), SIGMA, negate),
                        ) for index, seed in enumerate(wave)
                    ])
                    batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(prompts), self._dense_sampling_params_v4(0),
                            use_tqdm=False,
                        ) for index, _seed in enumerate(wave)
                    ])
                finally:
                    self._restore_all_engines_exact()
                if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
                    raise RuntimeError("v38a signed population wave is incomplete")
                for batch in batches:
                    score = _score_equal_unit(bundle, dense_items, batch)
                    signed[sign].append(score["equal_unit_mean"])
                    unweighted[sign].append(score["unweighted_row_mean"])
                    dense_hashes[sign].append(score["dense_result_sha256"])
        checks = self._rpc_all_engines_v4("verify_self_exact_reference", ())
        if any(item.get("passed") is not True for item in checks):
            raise RuntimeError("v38a post-population reference check failed")
        boundary = self._population_boundary_audit_v4(0)
        post_probe = self._base_probe_v38a(bundle, dense_items, prompts)
        if pre_probe != post_probe:
            raise RuntimeError("v38a base probe drifted across population")
        central = [
            0.5 * (plus - minus)
            for plus, minus in zip(signed["plus"], signed["minus"])
        ]
        coefficients, standardization = anchor_v13._standardize_v13(central)
        coefficient_identity = coefficient_sha256(SEEDS, coefficients)
        plan_id = canonical_sha256({
            "schema": "eggroll-es-distributed-plan-id-v38a",
            "iteration": 0,
            "coefficient_sha256": coefficient_identity,
            "reference_generation": self._v3_reference_generation,
            "reference_sha256": self._v3_reference_identity["sha256"],
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "reward_config_sha256": self._v4_reward_config_sha256,
            "runtime_mapping_sha256": canonical_sha256(self._v4_layer_plan_install),
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "train_bundle_content_sha256": bundle[
                "content_sha256_before_self_field"
            ],
        })
        plan = {
            "schema": "eggroll-es-equal-unit-seed-plan-v38a",
            "iteration": 0,
            "seeds": list(SEEDS),
            "coefficients": coefficients,
            "coefficient_sha256": coefficient_identity,
            "applied_alpha": 0.0,
            "applications": [],
            "response": {
                "equal_unit_sign_scores": signed,
                "unweighted_row_sign_scores": unweighted,
                "dense_result_sha256": dense_hashes,
                "central_response": central,
                "standardization": standardization,
            },
            "identity_audit": {
                "pre_probe_sha256": pre_probe,
                "post_probe_sha256": post_probe,
                "exact_reference_checks": checks,
                "passed": True,
            },
            "population_boundary_audit_v4": boundary,
            "dense_gold_reward_v4": {
                "config": dict(self._v4_reward_config),
                "reward_config_sha256": self._v4_reward_config_sha256,
            },
            "frozen_layer_plan_v4": {
                key: self._v4_layer_plan[key] for key in (
                    "path", "file_sha256", "plan_sha256", "model_config_path",
                    "model_config_sha256",
                )
            },
            "distributed_update_v4": {
                "schema": "eggroll-es-distributed-seed-plan-v4",
                "plan_id": plan_id,
                "engine_count": 4,
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
                    self._v4_layer_plan_install
                ),
                "population_boundary_audit_sha256": boundary["audit_sha256"],
            },
            "train_bundle": {
                "content_sha256": bundle["content_sha256_before_self_field"],
                "rows": 448, "conflict_units": 208,
                "weight_identity_sha256": bundle["weight_identity_sha256"],
            },
        }
        plan["frozen_layer_plan_v4"]["runtime_mapping"] = dict(
            self._v4_layer_plan_install
        )
        plan["frozen_layer_plan_v4"]["runtime_mapping_sha256"] = canonical_sha256(
            self._v4_layer_plan_install
        )
        self._latest_anchor_plan = plan
        self._v3_active_plan_id = plan_id
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._persist_anchor_plan(plan)
        anchor_v4.FrozenLayerDenseRewardMixinV4.apply_seed_coefficients(
            self, plan, ALPHA,
        )
        final_identity = plan["applications"][-1]["final_identity"]
        snapshots = self._rpc_all_engines_v4(
            "save_selected_snapshot_v38a",
            (str(Path(snapshot_path).resolve()), final_identity["sha256"], ALPHA),
        )
        written = [item for item in snapshots if item.get("written") is True]
        if (
            len(written) != 1
            or written[0].get("rank") != 0
            or any(item.get("final_identity") != final_identity for item in snapshots)
        ):
            raise RuntimeError("v38a selected snapshot consensus changed")
        return {
            "schema": "eggroll-es-equal-unit-update-v38a",
            "status": "one_nonzero_update_sealed_train_only",
            "sigma": SIGMA,
            "alpha": ALPHA,
            "population_size": 32,
            "signed_sequence_presentations": 32 * 2 * 448,
            "coefficient_sha256": coefficient_identity,
            "coefficients": coefficients,
            "standardization": standardization,
            "plan_id": plan_id,
            "application": plan["applications"][-1],
            "snapshot_reports": snapshots,
            "snapshot_file_sha256": written[0]["file_sha256"],
            "snapshot_file_bytes": written[0]["file_bytes"],
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
        }


def load_trainer(layer_plan_bundle):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    anchor_v13.validate_frozen_layer_plan_bundle_v13(layer_plan_bundle)
    parent = anchor_v13.anchor_v11c.load_trainer(layer_plan_bundle)

    class EqualUnitUpdateTrainerV38A(EqualUnitUpdateMixinV38A, parent):
        def launch_engines(
            self,
            num_engines=4,
            n_gpu_per_vllm_engine=1,
            model_name="Qwen/Qwen2.5-Math-1.5B",
            precision="bfloat16",
        ):
            if int(num_engines) != REQUIRED_ENGINE_COUNT:
                raise ValueError("v38a requires exactly four engines")
            if int(n_gpu_per_vllm_engine) != 1:
                raise ValueError("v38a requires TP=1")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )
            from es_at_scale.trainer.es_trainer import ESNcclLLM

            pgs = create_placement_groups_v38a(
                placement_group, num_engines,
            )
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

    return EqualUnitUpdateTrainerV38A
