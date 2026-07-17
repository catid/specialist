#!/usr/bin/env python3
"""Four-TP1 Qwen3.6 calibration for the V66 mirrored LoRA-ES protocol.

Import and ``--dry-run`` paths are CPU-only.  The live path requires an exact
self-hashed preregistration plus ``--execute``, four exclusive idle GPUs, the
sealed es-at-scale interpreter, and fresh artifact paths.  It opens only the
registered V434 train panel, evaluates eight +/- directions with common random
conditions, executes one nonzero V41A update candidate, and then exactly aborts
to the original FP32 master.  It never saves or promotes the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import eggroll_es_mirrored_v66 as mirrored


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66.json"
).resolve()
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66_lora_es_mirrored_crn_qwen36_calibration"
).resolve()
ATTEMPT = (
    RUN_DIR.parent / ".v66_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
).resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v66.jsonl").resolve()
POPULATION = (RUN_DIR / "mirrored_population_v66.json").resolve()
UPDATE = (RUN_DIR / "pair_difference_update_v66.json").resolve()
REPORT = (RUN_DIR / "mirrored_calibration_report_v66.json").resolve()
FAILURE = (RUN_DIR / "failure_v66.json").resolve()

WORKER_EXTENSION_V66 = (
    "eggroll_es_worker_lora_v66.LoRAAdapterStateWorkerExtensionV66"
)
REQUIRED_PYTHON_V66 = (ROOT / "es-at-scale/.venv/bin/python").absolute()
MASTER_SHA256_V66 = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)
MASTER_RUNTIME_SHA256_V66 = (
    "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
)
TRAIN_DATASET_SHA256_V66 = (
    "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a"
)
TRAIN_PANEL_FILE_SHA256_V66 = (
    "dd2c857b75617351d64cfce29f5a8e5d79ce9da212e4db50d22f2de3795c70a1"
)
TRAIN_PANEL_CONTENT_SHA256_V66 = (
    "cdfa9d10669171d5d814b55df1f674a89dfa557c5376b45c8d0073e5d1acaec7"
)
EVALUATION_SEED_V66 = 2_026_071_702


def file_sha256_v66(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifacts_v66() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "gpu_log": str(GPU_LOG),
        "population": str(POPULATION),
        "update": str(UPDATE),
        "report": str(REPORT),
        "failure": str(FAILURE),
    }


def _write_self_hashed_v66(path: Path, value: dict) -> dict:
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        mirrored.canonical_sha256_v66(result)
    )
    payload = (json.dumps(
        result,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def load_preregistration_v66(args) -> dict:
    import build_lora_es_mirrored_calibration_preregistration_v66 as builder

    path = Path(args.preregistration).resolve()
    if file_sha256_v66(path) != args.preregistration_sha256:
        raise RuntimeError("v66 preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or mirrored.canonical_sha256_v66(compact)
        != args.preregistration_content_sha256
        or value != builder.build_preregistration_v66()
        or value.get("schema")
        != "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66"
        or value.get("status")
        != "sealed_before_v66_train_semantics_model_ray_or_gpu_access"
        or value.get("artifacts") != artifacts_v66()
    ):
        raise RuntimeError("v66 preregistration content changed")
    return value


def parser_v66() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def _prepare_train_only_inputs_v66(trainer, prior, runtime52):
    panel = runtime52.load_generation_panel_v52()
    bundle = runtime52.load_train_bundle_v52()
    if (
        panel.get("content_sha256_before_self_field")
        != TRAIN_PANEL_CONTENT_SHA256_V66
        or file_sha256_v66(runtime52.design.TRAIN_GENERATION_PANEL_V52)
        != TRAIN_PANEL_FILE_SHA256_V66
        or bundle.get("dataset", {}).get("file_sha256")
        != TRAIN_DATASET_SHA256_V66
        or bundle.get("content_sha256_before_self_field")
        != runtime52.design.TRAIN_BUNDLE_CONTENT_SHA256_V52
    ):
        raise RuntimeError("v66 authorized train panel identity changed")
    rows = {
        row_sha: index for index, row_sha in enumerate(bundle["row_sha256"])
    }
    selected = panel["subset"]["items"]
    if (
        len(selected) != 64
        or any(item.get("request_index") != index
               for index, item in enumerate(selected))
        or len({item["unit_identity_sha256"] for item in selected}) != 64
    ):
        raise RuntimeError("v66 train-only panel coverage changed")
    indices = [rows[item["row_sha256"]] for item in selected]
    questions = [bundle["questions"][index] for index in indices]
    answers = [bundle["answers"][index] for index in indices]
    prompts = [
        prior.v40a.base.specialist_template(question)
        for question in questions
    ]
    dense_items = prior.anchor_v4.prepare_gold_answer_items_v4(
        trainer.tokenizer, prompts, answers
    )
    requests = [
        {"prompt_token_ids": item["prompt_token_ids"]}
        for item in dense_items
    ]
    prompt_sha = mirrored.canonical_sha256_v66(requests)
    judge = {
        "schema": "train-only-equal-conflict-unit-answer-logprob-v66",
        "reward_config_sha256": prior.anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
        "panel_content_sha256": TRAIN_PANEL_CONTENT_SHA256_V66,
        "request_order_sha256": panel["request_order_sha256"],
        "unit_identity_order_sha256": mirrored.canonical_sha256_v66([
            item["unit_identity_sha256"] for item in selected
        ]),
        "aggregation": "uniform mean of 64 conflict-unit representatives",
        "raw_questions_answers_or_outputs_persisted": False,
    }
    decode = {
        "n": 1,
        "seed": EVALUATION_SEED_V66,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "prompt_logprobs": 1,
        "detokenize": False,
    }
    payload = mirrored.common_evaluation_payload_v66(
        requests, decode, judge, EVALUATION_SEED_V66
    )
    if payload["contract"]["prompt_block_sha256"] != prompt_sha:
        raise RuntimeError("v66 tokenized prompt identity changed")
    return {
        "panel": panel,
        "dense_items": dense_items,
        "payload": payload,
        "answer_token_count_per_candidate": sum(
            item["answer_token_count"] for item in dense_items
        ),
        "input_receipt": {
            "schema": "v66-authorized-train-only-input-receipt",
            "dataset_file_sha256": TRAIN_DATASET_SHA256_V66,
            "panel_file_sha256": TRAIN_PANEL_FILE_SHA256_V66,
            "panel_content_sha256": TRAIN_PANEL_CONTENT_SHA256_V66,
            "rows": 64,
            "conflict_units": 64,
            "prompt_token_ids_sha256": prompt_sha,
            "evaluation_contract_sha256": payload["contract"][
                "evaluation_contract_sha256"
            ],
            "protected_dev_ood_or_holdout_opened": False,
            "raw_questions_answers_or_outputs_persisted": False,
        },
    }


@dataclass(frozen=True)
class _RuntimeHandleV66:
    kind: str
    ref: Any


class _RayMirroredCallbacksV66:
    def __init__(
        self,
        trainer,
        prior,
        prepared: dict,
        master_sha: str,
        phase,
    ) -> None:
        from vllm import SamplingParams

        self.trainer = trainer
        self.prior = prior
        self.prepared = prepared
        self.master_sha = master_sha
        self.phase = phase
        decode = prepared["payload"]["decode_contract"]
        self.sampling = SamplingParams(
            n=decode["n"],
            seed=decode["seed"],
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            max_tokens=decode["max_tokens"],
            prompt_logprobs=decode["prompt_logprobs"],
            detokenize=decode["detokenize"],
        )

    def submit_materialize(self, assignment: dict):
        self.phase.value = (
            f"mirrored_wave_{assignment['wave_index']}_materialize_all_actors"
        )
        ref = self.trainer.engines[
            assignment["engine_rank"]
        ].collective_rpc.remote(
            "materialize_mirrored_adapter_v66",
            args=(
                assignment["direction_seed"],
                assignment["sigma"],
                assignment["sign"],
                assignment["pair_id"],
                assignment["evaluation_contract_sha256"],
                self.master_sha,
            ),
        )
        return _RuntimeHandleV66("collective", ref)

    def submit_evaluate(self, assignment: dict, payload: dict):
        if payload["contract"]["evaluation_contract_sha256"] != (
            assignment["evaluation_contract_sha256"]
        ):
            raise RuntimeError("v66 evaluation payload changed at submission")
        self.phase.value = (
            f"mirrored_wave_{assignment['wave_index']}_generation_all_actors"
        )
        ref = self.trainer.engines[assignment["engine_rank"]].generate.remote(
            payload["prompts"],
            self.sampling,
            use_tqdm=False,
            lora_request=self.prior._lora_request(),
        )
        return _RuntimeHandleV66("generation", ref)

    def submit_restore(self, rank: int, reason: str):
        self.phase.value = f"{reason}_restore_all_actors"
        ref = self.trainer.engines[rank].collective_rpc.remote(
            "restore_mirrored_adapter_v66",
            args=(self.master_sha, reason),
        )
        return _RuntimeHandleV66("collective", ref)

    def resolve_one(self, handle: _RuntimeHandleV66):
        values = self.trainer._resolve([handle.ref])
        if len(values) != 1:
            raise RuntimeError("v66 Ray handle coverage changed")
        value = values[0]
        if handle.kind == "collective":
            if not isinstance(value, list) or len(value) != 1:
                raise RuntimeError("v66 collective worker coverage changed")
            return value[0]
        if handle.kind != "generation" or not isinstance(value, list):
            raise RuntimeError("v66 runtime handle kind changed")
        dense = self.prior.anchor_v4.score_gold_answer_outputs_v4(
            self.prepared["dense_items"], value
        )
        reward = float(dense["mean_example_mean_logprob"])
        if (
            dense["example_count"] != 64
            or dense["answer_token_count"]
            != self.prepared["answer_token_count_per_candidate"]
            or not math.isfinite(reward)
        ):
            raise RuntimeError("v66 answer-logprob reward coverage changed")
        return reward


def _rpc_all_v66(trainer, method: str, args=()) -> list[dict]:
    values = trainer._resolve([
        engine.collective_rpc.remote(method, args=args)
        for engine in trainer.engines
    ])
    if len(values) != 4 or any(
        not isinstance(value, list) or len(value) != 1 for value in values
    ):
        raise RuntimeError(f"v66 incomplete four-actor RPC: {method}")
    return [value[0] for value in values]


def _activate_install_and_certify_v66(trainer, prior, v40a, phase) -> dict:
    """Register the sole vLLM slot before installing canonical FP32 state."""
    request = prior._lora_request()
    request_id = getattr(request, "lora_int_id", None)
    if (
        getattr(request, "lora_name", None)
        != "matched_lora_initialization_v41b"
        or isinstance(request_id, bool)
        or request_id != 1
        or Path(getattr(request, "lora_path", "")).resolve()
        != Path(prior.STAGED).resolve()
    ):
        raise RuntimeError("v66 staged LoRA request identity changed")
    phase.value = "activate_v434_lora_slot_all_actors"
    activations = _rpc_all_v66(trainer, "add_lora", (request,))
    if activations != [True] * 4:
        raise RuntimeError("v66 four-actor LoRA activation failed")
    active_slots = _rpc_all_v66(
        trainer, "active_lora_slot_certificate_v66", (1,)
    )
    if (
        len(active_slots) != 4
        or any(
            item.get("schema") != "v66-active-lora-slot-certificate"
            or item.get("expected_lora_int_id") != 1
            or item.get("active_lora_ids") != [1]
            or item.get("active_manager_cache_lora_ids") != [1]
            or item.get("loaded_cpu_cache_lora_ids") != [1]
            or item.get("active_slot_index") != 0
            or item.get("canonical_state_write_performed") is not False
            for item in active_slots
        )
    ):
        raise RuntimeError("v66 four-actor active LoRA slot proof failed")
    phase.value = "install_canonical_v434_master_all_actors"
    installations = _rpc_all_v66(
        trainer,
        "install_adapter_state_v41a",
        (
            str(prior.SOURCE_WEIGHTS),
            str(prior.SOURCE_CONFIG),
            v40a.file_sha256(prior.SOURCE_WEIGHTS),
            v40a.file_sha256(prior.SOURCE_CONFIG),
        ),
    )
    certificates = _rpc_all_v66(
        trainer, "mirrored_adapter_state_certificate_v66"
    )
    if (
        {item["current_identity"]["sha256"] for item in certificates}
        != {MASTER_SHA256_V66}
        or {
            item["materialization"]["runtime_values_sha256"]
            for item in certificates
        }
        != {MASTER_RUNTIME_SHA256_V66}
    ):
        raise RuntimeError("v66 canonical V434 install consensus changed")
    return {
        "request": request,
        "activations": activations,
        "active_slots": active_slots,
        "installations": installations,
        "certificates": certificates,
    }


def _compact_population_v66(execution: dict) -> dict:
    materializations = [{
        key: receipt[key] for key in (
            "direction_seed", "sigma", "sign", "pair_id",
            "evaluation_contract_sha256", "noise_protocol",
            "master_unchanged", "exact_restore_required",
        )
    } | {
        "master_sha256": receipt["master_identity"]["sha256"],
        "candidate_sha256": receipt["candidate_identity"]["sha256"],
        "runtime_values_sha256": receipt["materialization"][
            "runtime_values_sha256"
        ],
    } for receipt in execution["materialization_receipts"]]
    restorations = [{
        "reason": receipt["reason"],
        "restored_master_sha256": receipt["restored_identity"]["sha256"],
        "runtime_values_sha256": receipt["materialization"][
            "runtime_values_sha256"
        ],
        "algebraic_bf16_restore_used": receipt[
            "algebraic_bf16_restore_used"
        ],
        "terminal_poisoned": receipt["terminal_poisoned"],
    } for receipt in execution["restore_receipts"]]
    return {
        "schema": "v66-mirrored-qwen36-population-evidence",
        "plan_sha256": execution["plan_sha256"],
        "evaluation_contract_sha256": execution[
            "evaluation_contract_sha256"
        ],
        "signed_rewards": execution["signed_rewards"],
        "materializations": materializations,
        "restorations": restorations,
        "signed_reward_sha256": mirrored.canonical_sha256_v66(
            execution["signed_rewards"]
        ),
        "all_submitted_work_drained": execution[
            "all_submitted_work_drained"
        ],
        "all_four_actors_restored_after_every_wave": execution[
            "all_four_actors_restored_after_every_wave"
        ],
        "raw_questions_answers_or_outputs_persisted": False,
        "protected_dev_ood_or_holdout_opened": False,
    }


def _execute_and_abort_nonzero_update_v66(
    trainer,
    update: dict,
    master_sha: str,
    reference_generation: int,
    phase,
) -> dict:
    coefficient_l2 = math.sqrt(math.fsum(
        value * value for value in update["coefficients"]
    ))
    nonzero_pairs = sum(value != 0.0 for value in update["coefficients"])
    if coefficient_l2 == 0.0 or nonzero_pairs == 0:
        raise RuntimeError("v66 Qwen calibration produced a zero pair update")
    prepared = executed = aborts = None
    manifest_sha = None
    primary_error = None
    try:
        phase.value = "pair_difference_update_prepare_all_actors"
        prepared = _rpc_all_v66(
            trainer,
            "prepare_sharded_adapter_update_v41a",
            (
                update["direction_seeds"],
                update["coefficients"],
                update["coefficient_sha256"],
                update["worker_population_size"],
                4,
                update["worker_alpha"],
                "v66-qwen36-nonzero-calibration",
                master_sha,
                reference_generation,
            ),
        )
        manifests = {item["manifest_sha256"] for item in prepared}
        if (
            len(manifests) != 1
            or {item["rank"] for item in prepared} != {0, 1, 2, 3}
        ):
            raise RuntimeError("v66 update prepare consensus changed")
        manifest_sha = next(iter(manifests))
        phase.value = "pair_difference_update_execute_all_actors"
        executed = _rpc_all_v66(
            trainer,
            "execute_sharded_adapter_update_v41a",
            (manifest_sha,),
        )
        candidate_hashes = {
            item["candidate_identity"]["sha256"] for item in executed
        }
        runtime_hashes = {
            item["materialization"]["runtime_values_sha256"]
            for item in executed
        }
        if (
            len(candidate_hashes) != 1
            or len(runtime_hashes) != 1
            or candidate_hashes == {master_sha}
            or runtime_hashes == {MASTER_RUNTIME_SHA256_V66}
            or any(item["master_committed"] is not False for item in executed)
        ):
            raise RuntimeError("v66 nonzero update candidate consensus changed")
    except BaseException as error:
        primary_error = error
    finally:
        try:
            phase.value = "pair_difference_update_abort_all_actors"
            aborts = _rpc_all_v66(
                trainer,
                "abort_mirrored_update_if_present_v66",
                (
                    master_sha,
                    "v66_nonzero_calibration_no_commit",
                    manifest_sha,
                ),
            )
        except BaseException as abort_error:
            raise RuntimeError(
                "v66 could not prove four-actor update abort"
            ) from abort_error
    if primary_error is not None:
        raise primary_error
    if any(
        item["restored_identity"]["sha256"] != master_sha
        or item["terminal_poisoned"] is not False
        for item in aborts
    ):
        raise RuntimeError("v66 update abort master consensus changed")
    return {
        "schema": "v66-nonzero-qwen36-pair-difference-update-receipt",
        "coefficient_l2": coefficient_l2,
        "nonzero_pair_differences": nonzero_pairs,
        "direction_count": len(update["coefficients"]),
        "worker_alpha": update["worker_alpha"],
        "worker_population_size": update["worker_population_size"],
        "effective_noise_scale": update["effective_noise_scale"],
        "coefficient_sha256": update["coefficient_sha256"],
        "manifest_sha256": manifest_sha,
        "prepared_rank_shards": [{
            "rank": item["rank"],
            "shard_indices": item["shard_indices"],
            "shard_seeds": item["shard_seeds"],
        } for item in prepared],
        "candidate_master_sha256": executed[0]["candidate_identity"]["sha256"],
        "candidate_runtime_values_sha256": executed[0]["materialization"][
            "runtime_values_sha256"
        ],
        "candidate_differs_from_master": True,
        "candidate_runtime_differs_from_master": True,
        "master_committed": False,
        "all_four_abort_receipts_exact": True,
        "checkpoint_snapshot_or_promotion_performed": False,
    }


def _gpu_wave_summary_v66(path: Path, expected_pids: dict[int, int]) -> dict:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_wave = {}
    for wave in range(4):
        phase = f"mirrored_wave_{wave}_generation_all_actors"
        by_gpu = {}
        for gpu in range(4):
            selected = [
                row for row in rows
                if row["phase"] == phase and row["gpu"] == gpu
                and expected_pids[gpu] in row["compute_pids"]
            ]
            positive = [
                row for row in selected if row["utilization_percent"] > 0
            ]
            if not selected or not positive:
                raise RuntimeError(
                    f"v66 GPU {gpu} lacked useful activity in mirrored wave {wave}"
                )
            by_gpu[str(gpu)] = {
                "resident_samples": len(selected),
                "positive_resident_samples": len(positive),
                "peak_utilization_percent": max(
                    row["utilization_percent"] for row in selected
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in selected
                ),
            }
        by_wave[str(wave)] = by_gpu
    return {
        "schema": "v66-four-gpu-mirrored-wave-activity",
        "waves": 4,
        "all_four_physical_gpus_positive_in_every_wave": True,
        "by_wave": by_wave,
    }


def execute_v66(preregistration: dict, args) -> int:
    import run_lora_es_generation_boundary_v48b as v48b
    import run_lora_es_nested_population_v52 as runtime52
    import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64

    prior = v48b.v43i
    v40a = prior.v40a
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v66 requires fresh attempt and run artifact paths")
    expectation = runtime64.base_model_artifact_expectation_v64()
    pre_model = runtime64.verify_base_model_artifacts_v64(expectation)
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "v66-mirrored-qwen36-calibration-attempt",
        "status": "launching_train_only_nonpromoting_calibration",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "base_model_artifact_receipt": pre_model,
        "preflight": preflight,
        "protected_dev_ood_or_holdout_opened": False,
        "checkpoint_snapshot_or_promotion_authorized": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = saved = monitor = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    cleanup = idle = None
    resident_started_ns = resident_ended_ns = None
    started = time.monotonic()
    try:
        panel = runtime52.load_generation_panel_v52()
        v48b._SEALED_SUBSET = panel
        with v48b.patched_v43i_v48b(), runtime52.patched_runtime_v52(prior):
            prior.WORKER_EXTENSION = WORKER_EXTENSION_V66
            runtime52.verify_adapter_contract_v52()
            v40a.base.set_seed(EVALUATION_SEED_V66)
            resident_started_ns = time.monotonic_ns()
            trainer, saved = prior._make_trainer(preregistration)
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote()
                for engine in trainer.engines
            ])
            worker_ids = _rpc_all_v66(trainer, "runtime_identity_v40a")
            pid_map = v40a.validate_identities(actor_ids, worker_ids)
            monitor = threading.Thread(
                target=v40a.monitor_gpus,
                args=(stop, phase, pid_map, GPU_LOG, failures),
                daemon=True,
            )
            monitor.start()
            prepared_inputs = _prepare_train_only_inputs_v66(
                trainer, prior, runtime52
            )
            recipe = preregistration["fixed_recipe"]
            plan = mirrored.mirrored_population_plan_v66(
                recipe["direction_seeds"],
                recipe["sigma"],
                prepared_inputs["payload"],
            )
            activation = _activate_install_and_certify_v66(
                trainer, prior, v40a, phase
            )
            installations = activation["installations"]
            references = _rpc_all_v66(
                trainer, "capture_adapter_reference_v41a"
            )
            reference_generations = {
                item["reference_generation"] for item in references
            }
            if len(reference_generations) != 1:
                raise RuntimeError("v66 reference generation consensus changed")
            callbacks = _RayMirroredCallbacksV66(
                trainer,
                prior,
                prepared_inputs,
                MASTER_SHA256_V66,
                phase,
            )
            execution = mirrored.execute_mirrored_plan_v66(
                plan,
                prepared_inputs["payload"],
                callbacks.submit_materialize,
                callbacks.submit_evaluate,
                callbacks.submit_restore,
                callbacks.resolve_one,
            )
            population_value = _compact_population_v66(execution)
            population_value.update({
                "input_receipt": prepared_inputs["input_receipt"],
                "plan": plan,
                "install_master_consensus_sha256": MASTER_SHA256_V66,
                "installation_count": len(installations),
            })
            population_artifact = _write_self_hashed_v66(
                POPULATION, population_value
            )
            update = mirrored.pair_difference_update_v66(
                plan,
                execution["signed_rewards"],
                recipe["learning_rate"],
            )
            update_receipt = _execute_and_abort_nonzero_update_v66(
                trainer,
                update,
                MASTER_SHA256_V66,
                next(iter(reference_generations)),
                phase,
            )
            update_artifact = _write_self_hashed_v66(UPDATE, {
                **update_receipt,
                "plan_sha256": plan["plan_sha256"],
                "population_content_sha256": population_artifact[
                    "content_sha256_before_self_field"
                ],
                "protected_dev_ood_or_holdout_opened": False,
            })
            final_certificates = _rpc_all_v66(
                trainer, "mirrored_adapter_state_certificate_v66"
            )
            if any(
                item["current_identity"]["sha256"] != MASTER_SHA256_V66
                or item["materialization"]["runtime_values_sha256"]
                != MASTER_RUNTIME_SHA256_V66
                or item["terminal_poisoned"] is not False
                for item in final_certificates
            ):
                raise RuntimeError("v66 final exact master consensus changed")
            stop.set()
            monitor.join(timeout=10)
            if monitor.is_alive() or not failures.empty():
                raise RuntimeError("v66 GPU monitor failed") from (
                    failures.get() if not failures.empty() else None
                )
            gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
            gpu_waves = _gpu_wave_summary_v66(GPU_LOG, pid_map)
            cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
            trainer = None
            resident_ended_ns = time.monotonic_ns()
            import ray
            ray.shutdown()
            idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        post_model = runtime64.verify_base_model_artifacts_v64(expectation)
        if post_model != pre_model:
            raise RuntimeError("v66 base-model bytes changed during calibration")
        resident_seconds = (
            resident_ended_ns - resident_started_ns
        ) / 1_000_000_000
        report = _write_self_hashed_v66(REPORT, {
            "schema": "v66-mirrored-crn-qwen36-calibration-report",
            "status": "complete_nonzero_train_only_no_commit",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "population": {
                "path": str(POPULATION),
                "file_sha256": file_sha256_v66(POPULATION),
                "content_sha256": population_artifact[
                    "content_sha256_before_self_field"
                ],
            },
            "nonzero_update": {
                "path": str(UPDATE),
                "file_sha256": file_sha256_v66(UPDATE),
                "content_sha256": update_artifact[
                    "content_sha256_before_self_field"
                ],
                "coefficient_l2": update_artifact["coefficient_l2"],
                "nonzero_pair_differences": update_artifact[
                    "nonzero_pair_differences"
                ],
                "candidate_differs_from_master": True,
                "candidate_runtime_differs_from_master": True,
                "master_committed": False,
            },
            "gpu_activity": gpu,
            "gpu_waves": gpu_waves,
            "gpu_log_file_sha256": file_sha256_v66(GPU_LOG),
            "compute_ledger": {
                "physical_gpus": [0, 1, 2, 3],
                "model_resident_seconds_per_gpu": resident_seconds,
                "charged_gpu_seconds": 4 * resident_seconds,
                "optimization_generated_rollouts": plan[
                    "signed_population_size"
                ] * prepared_inputs["input_receipt"]["rows"],
                "generated_tokens_upper_bound": plan[
                    "signed_population_size"
                ] * prepared_inputs["input_receipt"]["rows"],
                "teacher_forced_answer_tokens": plan[
                    "signed_population_size"
                ] * prepared_inputs["answer_token_count_per_candidate"],
                "checkpoint_count": 0,
            },
            "base_model_prelaunch_artifact_receipt": pre_model,
            "base_model_postrun_artifact_receipt": post_model,
            "final_master_sha256": MASTER_SHA256_V66,
            "final_runtime_values_sha256": MASTER_RUNTIME_SHA256_V66,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "raw_questions_answers_or_outputs_persisted": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_performed": False,
        })
        print(json.dumps({
            "report": str(REPORT),
            "report_file_sha256": file_sha256_v66(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "nonzero_pair_differences": report["nonzero_update"][
                "nonzero_pair_differences"
            ],
            "all_four_physical_gpus_positive_in_every_wave": True,
            "master_committed": False,
            "final_exact_master_restored": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        cleanup_failure = None
        if trainer is not None:
            try:
                cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
                trainer = None
                import ray
                ray.shutdown()
                idle = v40a.cleanup_v38a.wait_for_gpu_idle()
            except BaseException as cleanup_error:
                cleanup_failure = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
        try:
            import ray
            ray.shutdown()
            if idle is None:
                idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        except BaseException as shutdown_error:
            if cleanup_failure is None:
                cleanup_failure = {
                    "type": type(shutdown_error).__name__,
                    "message": str(shutdown_error),
                }
        failed_resident_end_ns = time.monotonic_ns()
        failed_ledger = None
        if resident_started_ns is not None:
            seconds = (
                failed_resident_end_ns - resident_started_ns
            ) / 1_000_000_000
            failed_ledger = {
                "model_allocation_or_residency_seconds_per_gpu": seconds,
                "charged_gpu_seconds": 4 * seconds,
                "includes_failed_observed_work": True,
            }
        _write_self_hashed_v66(FAILURE, {
            "schema": "v66-mirrored-qwen36-calibration-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "cleanup": cleanup,
            "cleanup_failure": cleanup_failure,
            "final_gpu_idle": idle,
            "compute_ledger": failed_ledger,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_performed": False,
        })
        raise
    finally:
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


def main(argv=None) -> int:
    args = parser_v66().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v66 requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v66(args)
    if args.dry_run:
        recipe = preregistration["fixed_recipe"]
        print(json.dumps({
            "schema": preregistration["schema"],
            "model": "Qwen3.6-35B-A3B",
            "four_tp1_engines": True,
            "direction_count": len(recipe["direction_seeds"]),
            "signed_population_size": 2 * len(recipe["direction_seeds"]),
            "train_only_rows_per_candidate": 64,
            "expected_artifacts": artifacts_v66(),
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_authorized": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != REQUIRED_PYTHON_V66:
        raise RuntimeError(
            f"v66 requires {REQUIRED_PYTHON_V66}; observed {sys.executable}"
        )
    return execute_v66(preregistration, args)


if __name__ == "__main__":
    raise SystemExit(main())
