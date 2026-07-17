#!/usr/bin/env python3
"""Build a content-addressed CPU preview of the CPT/SFT/ES ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

import build_train_shadow_folds_v37a as conflict_units
import curriculum_ablation_v1 as curriculum
import qa_quality
import train_sampling_ablation_v1 as sampling


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "cpt_sft_es_curriculum_ablation_v1.json"
).resolve()
EVALUATION_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
SAMPLING_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "recipe_sampling_ablation_v1.json"
).resolve()
CORPUS_REGISTRY = (
    ROOT / "data/site_corpora/registry/site_corpus_registry_v1.json"
).resolve()
SFT_SOURCE = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/train_v434_fold3_v49d.jsonl"
).resolve()
TOKENIZER = (ROOT / "models/Qwen3.6-35B-A3B/tokenizer.json").resolve()
TOKENIZER_CONFIG = (
    ROOT / "models/Qwen3.6-35B-A3B/tokenizer_config.json"
).resolve()
MODEL_CONFIG = (ROOT / "models/Qwen3.6-35B-A3B/config.json").resolve()
MODEL_INDEX = (
    ROOT / "models/Qwen3.6-35B-A3B/model.safetensors.index.json"
).resolve()
INITIAL_ADAPTER = (
    ROOT / "experiments/eggroll_es_hpo/initial_adapters/"
    "matched_lora_initialization_v41a_seed20260715041"
).resolve()

EXPECTED = {
    "evaluation_file": "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf",
    "evaluation_content": "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad",
    "sampling_file": "8fe9005c532dbdb286edfd359cbbaf689c46f529b3debb3edc74f48e5787b301",
    "sampling_content": "18cab815193d05de6e7416b17c1ffeae334a6a613f3899faa459cc719144e97f",
    "registry_file": "633b191178ee378dcd058b56d2a58b7a76d6e05daad154a4eb93c619fc8ee06f",
    "train_file": "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a",
    "tokenizer": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "model_config": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "model_index": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    "initial_weights": "29fe0beead8a491cf06e9f562a1838d9c44e94a74e6a4024549e87f10557111f",
    "initial_config": "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52",
    "initial_manifest": "a2fa79e6ac06f75743d3fee8f5c0b1aabe6bb83b52b05910ed6460438e2640a2",
    "initial_state": "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6",
    "adapter_surface": "6c4c219f92fba3d7d01e08f439b7b1f21a1d07bc9893cdd18f860994668e0fb8",
}
MASTER_SEED = "specialist-0j5.5-cpt-sft-es-20260717"
CPT_SOURCE_TOKEN_CAP = 4_845
CPT_TARGET_TOKENS = 99_216
CPT_CHUNK_TOKEN_CAP = 1_024


def file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_v1(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"curriculum expected object: {path}")
    return value


def _read_jsonl_v1(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"curriculum expected nonempty JSONL objects: {path}")
    return rows


def _sealed_parent_v1(path: Path, file_sha: str, content_sha: str) -> dict:
    if file_sha256_v1(path) != file_sha:
        raise RuntimeError(f"curriculum sealed parent file changed: {path}")
    value = _read_json_v1(path)
    compact = dict(value)
    observed = compact.pop("content_sha256_before_self_field", None)
    if observed != content_sha or curriculum.canonical_sha256_v1(compact) != observed:
        raise RuntimeError(f"curriculum sealed parent content changed: {path}")
    return value


def _key_v1(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(map(str, values)).encode("utf-8")
    ).hexdigest()


def _token_id_sha256_v1(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise RuntimeError("curriculum tokenizer emitted an invalid token ID")
        digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _with_content_sha_v1(value: dict) -> dict:
    result = dict(value)
    result["content_sha256"] = curriculum.canonical_sha256_v1(result)
    return result


def build_cpt_input_v1(tokenizer: Tokenizer) -> dict:
    if file_sha256_v1(CORPUS_REGISTRY) != EXPECTED["registry_file"]:
        raise RuntimeError("curriculum Markdown registry changed")
    registry = _read_json_v1(CORPUS_REGISTRY)
    if (
        registry.get("schema") != "site-corpus-registry-v1"
        or registry.get("tokenizer", {}).get("tokenizer_json_sha256")
        != EXPECTED["tokenizer"]
        or registry.get("totals", {}).get("artifact_count") != 33
    ):
        raise RuntimeError("curriculum Markdown registry contract changed")
    eligible = [
        artifact for artifact in registry["artifacts"]
        if artifact.get("declared_direct_training_ready") is True
        and artifact.get("rights_basis", {}).get("status")
        in curriculum.ALLOWED_RIGHTS_V1
        and str(artifact.get("rights_basis", {}).get("promotion_gate", ""))
        .startswith("ready_")
    ]
    if len(eligible) != 29:
        raise RuntimeError("curriculum rights-promoted Markdown inventory changed")
    token_inventory = {}
    allocations = {}
    for artifact in eligible:
        path = (ROOT / artifact["markdown_path"]).resolve()
        if file_sha256_v1(path) != artifact["markdown_sha256"]:
            raise RuntimeError("curriculum Markdown artifact bytes changed")
        token_ids = tokenizer.encode(
            path.read_text(encoding="utf-8"), add_special_tokens=False
        ).ids
        if len(token_ids) != artifact["qwen36_token_count"]:
            raise RuntimeError("curriculum Markdown registry token count changed")
        token_inventory[artifact["resource_id"]] = token_ids
        allocations[artifact["resource_id"]] = min(
            CPT_SOURCE_TOKEN_CAP, len(token_ids)
        )
    excess = sum(allocations.values()) - CPT_TARGET_TOKENS
    if excess != 3:
        raise RuntimeError("curriculum CPT cap/target arithmetic changed")
    trim_resource = max(
        allocations,
        key=lambda resource: _key_v1(MASTER_SEED, "global-trim", resource),
    )
    if allocations[trim_resource] <= excess:
        raise RuntimeError("curriculum deterministic CPT trim is invalid")
    allocations[trim_resource] -= excess

    units = []
    by_resource = {artifact["resource_id"]: artifact for artifact in eligible}
    for resource in sorted(by_resource):
        artifact = by_resource[resource]
        token_ids = token_inventory[resource]
        selected = allocations[resource]
        maximum_start = len(token_ids) - selected
        start = (
            int(_key_v1(MASTER_SEED, "contiguous-window", resource)[:16], 16)
            % (maximum_start + 1)
            if maximum_start else 0
        )
        stop = start + selected
        units.append({
            "resource_id": resource,
            "source_document_identity_sha256": artifact[
                "source_document_identity_sha256"
            ],
            "markdown_path": artifact["markdown_path"],
            "markdown_sha256": artifact["markdown_sha256"],
            "available_tokens": len(token_ids),
            "selected_tokens": selected,
            "token_start": start,
            "token_stop": stop,
            "selected_token_ids_sha256": _token_id_sha256_v1(
                token_ids[start:stop]
            ),
            "rights_status": artifact["rights_basis"]["status"],
            "promotion_gate": artifact["rights_basis"]["promotion_gate"],
            "direct_training_ready": True,
            "training_role": "cpt_raw_markdown",
            "training_format": "causal_next_token_markdown",
            "assistant_supervision": False,
        })
    excluded = [{
        "resource_id": artifact["resource_id"],
        "available_tokens": artifact["qwen36_token_count"],
        "rights_status": artifact["rights_basis"]["status"],
        "promotion_gate": artifact["rights_basis"]["promotion_gate"],
        "reason": "rights_promotion_gate_not_ready",
    } for artifact in registry["artifacts"] if artifact not in eligible]
    result = {
        "schema": "specialist-cpt-raw-markdown-input-v1",
        "training_role": "causal_language_modeling",
        "assistant_supervision": False,
        "registry_path": str(CORPUS_REGISTRY),
        "registry_file_sha256": EXPECTED["registry_file"],
        "tokenizer_json_sha256": EXPECTED["tokenizer"],
        "selection_seed": MASTER_SEED,
        "selection_rule": (
            "one contiguous tokenizer window per source; cap at 4845; "
            "deterministically trim three tokens from one source to exact total"
        ),
        "source_token_cap": CPT_SOURCE_TOKEN_CAP,
        "maximum_source_token_fraction": CPT_SOURCE_TOKEN_CAP / CPT_TARGET_TOKENS,
        "chunk_token_cap": CPT_CHUNK_TOKEN_CAP,
        "cross_document_packing": False,
        "target_nonpadding_tokens": CPT_TARGET_TOKENS,
        "selected_nonpadding_tokens": sum(
            unit["selected_tokens"] for unit in units
        ),
        "eligible_source_documents": len(units),
        "units": units,
        "excluded_registry_artifacts": excluded,
        "policy_excluded_nontraining_manifests": registry[
            "excluded_nontraining_manifests"
        ],
    }
    return _with_content_sha_v1(result)


def _specialist_prompt_v1(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Answer the question briefly and factually. Return only the answer."
        "<|im_end|>\n<|im_start|>user\n"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def build_sft_input_v1(tokenizer: Tokenizer) -> dict:
    if file_sha256_v1(SFT_SOURCE) != EXPECTED["train_file"]:
        raise RuntimeError("curriculum SFT source bytes changed")
    source_rows = _read_jsonl_v1(SFT_SOURCE)
    if len(source_rows) != 448:
        raise RuntimeError("curriculum SFT source row count changed")
    reasons = Counter()
    rows = []
    for row in source_rows:
        reason = sampling.exclusion_reason(row)
        if reason is None:
            rows.append(row)
        else:
            reasons[reason] += 1
    if len(rows) != 433 or reasons != {"canonical_url_trivia": 15}:
        raise RuntimeError("curriculum verified SFT eligibility changed")
    conflict_inventory = conflict_units.build_conflict_units(rows)
    by_row = {}
    for unit in conflict_inventory:
        for index in unit["indices"]:
            if index in by_row:
                raise RuntimeError("curriculum SFT row entered two conflict units")
            by_row[index] = (
                unit["identity_sha256"], len(unit["indices"])
            )
    if len(by_row) != len(rows) or len(conflict_inventory) != 210:
        raise RuntimeError("curriculum SFT conflict-unit inventory changed")
    units = []
    prompt_total = answer_total = 0
    forbidden_protocol = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    for index, row in enumerate(rows):
        pair = qa_quality.qa_pair_from_record(row)
        if pair is None:
            raise RuntimeError("curriculum SFT verified row lost its QA pair")
        question, answer = pair
        if any(marker in answer for marker in forbidden_protocol):
            raise RuntimeError("curriculum SFT assistant answer contains protocol text")
        prompt_ids = tokenizer.encode(
            _specialist_prompt_v1(question), add_special_tokens=False
        ).ids
        all_ids = tokenizer.encode(
            _specialist_prompt_v1(question) + answer,
            add_special_tokens=False,
        ).ids
        if all_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("curriculum SFT prompt/answer boundary changed")
        answer_ids = all_ids[len(prompt_ids):]
        if not answer_ids or len(answer_ids) > 128 or len(all_ids) > 1_024:
            raise RuntimeError("curriculum SFT answer/sequence cap changed")
        conflict_id, conflict_rows = by_row[index]
        units.append({
            "fact_id": row["fact_id"],
            "row_sha256": conflict_units.row_sha256(row),
            "document_sha256": row["document_sha256"],
            "source": row["source"],
            "kind": row["kind"],
            "quality_schema": row["quality_schema"],
            "conflict_unit_identity_sha256": conflict_id,
            "conflict_unit_rows": conflict_rows,
            "prompt_tokens": len(prompt_ids),
            "answer_tokens": len(answer_ids),
            "verified_instructional": True,
            "url_trivia": False,
            "raw_markdown": False,
            "training_role": "sft_verified_instruction",
            "training_format": "masked_prompt_qa_chat",
        })
        prompt_total += len(prompt_ids)
        answer_total += len(answer_ids)
    result = {
        "schema": "specialist-verified-instruction-sft-input-v1",
        "training_role": "assistant_only_supervision",
        "raw_markdown_as_assistant_answer": False,
        "source_path": str(SFT_SOURCE),
        "source_file_sha256": EXPECTED["train_file"],
        "source_rows": len(source_rows),
        "eligible_rows": len(units),
        "excluded_rows": sum(reasons.values()),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "conflict_units": len(conflict_inventory),
        "base_pass_prompt_tokens": prompt_total,
        "base_pass_answer_tokens": answer_total,
        "base_pass_nonpadding_tokens": prompt_total + answer_total,
        "maximum_answer_tokens": 128,
        "maximum_sequence_tokens": 1_024,
        "prompt_mode": "es_exact_masked_prompt",
        "eos_appended": False,
        "unit_objective": "equal total mass per conservative conflict unit",
        "unit_objective_multiplier_cap": 3 / 2,
        "source_multiplier_min": 2 / 3,
        "source_multiplier_max": 3 / 2,
        "units": units,
    }
    if (prompt_total, answer_total) != (21_634, 11_438):
        raise RuntimeError("curriculum SFT token identity changed")
    return _with_content_sha_v1(result)


def build_es_input_v1() -> dict:
    sampling_contract = _sealed_parent_v1(
        SAMPLING_CONTRACT,
        EXPECTED["sampling_file"],
        EXPECTED["sampling_content"],
    )
    matches = [
        item for item in sampling_contract["variants"]
        if item.get("name") == "category_stratified"
    ]
    if len(matches) != 1:
        raise RuntimeError("curriculum category-stratified ES panel changed")
    variant = matches[0]
    items = [{
        "unit_identity_sha256": item["unit_identity_sha256"],
        "row_sha256": item["row_sha256"],
        "source": item["source"],
        "category": item["category"],
        "position": item["position"],
    } for item in variant["items"]]
    result = {
        "schema": "specialist-mirrored-es-curriculum-input-v1",
        "training_role": "train_reward_optimization",
        "sampling_contract_path": str(SAMPLING_CONTRACT),
        "sampling_contract_file_sha256": EXPECTED["sampling_file"],
        "sampling_contract_content_sha256": EXPECTED["sampling_content"],
        "variant": "category_stratified",
        "panel_identity_sha256": variant["ordered_unit_identity_sha256"],
        "rows_per_update": 64,
        "source_row_cap": 15,
        "directions": 8,
        "signed_population": 16,
        "rollouts_per_update": 1_024,
        "reward_surface": "registered_train_only_dense_gold_answer_logprob",
        "mirrored_pair_common_random_numbers": True,
        "items": items,
    }
    return _with_content_sha_v1(result)


def _stage_v1(
    stage_id: str,
    dataset_sha: str,
    target_gpu_seconds: int,
    updates: int,
    tokens: int,
    rollouts: int,
    index: int,
) -> dict:
    return {
        "stage_id": stage_id,
        "dataset_content_sha256": dataset_sha,
        "target_gpu_seconds": target_gpu_seconds,
        "optimizer_updates": updates,
        "nonpadding_tokens": tokens,
        "generated_rollouts": rollouts,
        "checkpoint_input": (
            "initial_adapter" if index == 0 else "previous_stage_output"
        ),
        "transition_mode": (
            "fresh_from_initial" if index == 0
            else "weights_only_new_stage_optimizer"
        ),
    }


def _arms_v1(dataset_hashes: dict[str, str]) -> list[dict]:
    definitions = [
        ("cpt_sft_es", (("cpt", 3_600, 96, 99_216, 0),
                        ("sft", 3_600, 48, 99_216, 0),
                        ("es", 7_200, 2, 0, 2_048))),
        ("direct_sft_es", (("sft", 7_200, 96, 198_432, 0),
                           ("es", 7_200, 2, 0, 2_048))),
        ("direct_sft", (("sft", 14_400, 192, 396_864, 0),)),
        ("direct_es", (("es", 14_400, 4, 0, 4_096),)),
    ]
    return [{
        "arm_id": arm_id,
        "total_target_gpu_seconds": 14_400,
        "stages": [
            _stage_v1(stage, dataset_hashes[stage], gpu, updates, tokens,
                      rollouts, index)
            for index, (stage, gpu, updates, tokens, rollouts)
            in enumerate(stages)
        ],
    } for arm_id, stages in definitions]


def _implementation_bindings_v1() -> dict:
    paths = {
        "builder": Path(__file__).resolve(),
        "validator": Path(curriculum.__file__).resolve(),
        "quality_rules": Path(qa_quality.__file__).resolve(),
        "conflict_units": Path(conflict_units.__file__).resolve(),
        "sampling_builder": Path(sampling.__file__).resolve(),
    }
    return {
        label: {"path": str(path), "file_sha256": file_sha256_v1(path)}
        for label, path in paths.items()
    }


def build_preregistration_v1() -> dict:
    for path, expected in (
        (TOKENIZER, EXPECTED["tokenizer"]),
        (TOKENIZER_CONFIG, EXPECTED["tokenizer_config"]),
        (MODEL_CONFIG, EXPECTED["model_config"]),
        (MODEL_INDEX, EXPECTED["model_index"]),
        (INITIAL_ADAPTER / "adapter_model.safetensors", EXPECTED["initial_weights"]),
        (INITIAL_ADAPTER / "adapter_config.json", EXPECTED["initial_config"]),
        (
            INITIAL_ADAPTER / "initialization_manifest_v41a.json",
            EXPECTED["initial_manifest"],
        ),
    ):
        if file_sha256_v1(path) != expected:
            raise RuntimeError(f"curriculum sealed model/adapter input changed: {path}")
    evaluation = _sealed_parent_v1(
        EVALUATION_CONTRACT,
        EXPECTED["evaluation_file"],
        EXPECTED["evaluation_content"],
    )
    protected = evaluation["roles"]["protected_holdout"]
    if (
        evaluation.get("disjointness", {}).get("passed") is not True
        or protected.get("access_authorized_by_this_contract") is not False
    ):
        raise RuntimeError("curriculum parent evaluation firewall changed")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    cpt = build_cpt_input_v1(tokenizer)
    sft = build_sft_input_v1(tokenizer)
    es = build_es_input_v1()
    datasets = {"cpt": cpt, "sft": sft, "es": es}
    dataset_hashes = {
        stage: value["content_sha256"] for stage, value in datasets.items()
    }
    result = {
        "schema": curriculum.PLAN_SCHEMA_V1,
        "status": "sealed_cpu_preview_launch_ineligible",
        "purpose": (
            "Ablate rights-promoted raw-Markdown CPT, verified instructional "
            "SFT, and mirrored LoRA-ES against direct SFT/ES controls under "
            "equal charged GPU seconds and source-disjoint evaluation."
        ),
        "evaluation_contract": {
            "path": str(EVALUATION_CONTRACT),
            "file_sha256": EXPECTED["evaluation_file"],
            "content_sha256": EXPECTED["evaluation_content"],
            "disjointness_passed": True,
            "protected_access_authorized": False,
            "protected_selection_identity_sha256": protected[
                "selected_identity_set_sha256"
            ],
            "current_adaptation_allowlist": [EXPECTED["train_file"]],
            "new_curriculum_inputs_currently_authorized": False,
        },
        "source_disjoint_extension": {
            "schema": curriculum.EXTENSION_SCHEMA_V1,
            "status": "required_not_built",
            "parent_evaluation_contract_content_sha256": EXPECTED[
                "evaluation_content"
            ],
            "protected_selection_identity_sha256": protected[
                "selected_identity_set_sha256"
            ],
            "identity_domains": list(curriculum.IDENTITY_DOMAINS_V1),
            "required_new_input_content_sha256s": sorted(dataset_hashes.values()),
            "audit_file_sha256": None,
            "audit_content_sha256": None,
            "cross_role_collision_counts": None,
            "audit_passed": False,
            "launch_eligible": False,
            "protected_semantics_opened": False,
            "protected_content_persisted": False,
        },
        "authorization": {
            "gpu_launch": False,
            "new_adaptation_inputs": False,
            "protected_holdout_access": False,
            "protected_holdout_selection_or_tuning": False,
            "future_launch_requires_disjoint_extension_and_gpu_throughput_calibration": True,
        },
        "model_and_adapter": {
            "model": "Qwen3.6-35B-A3B",
            "model_config_file_sha256": EXPECTED["model_config"],
            "model_index_file_sha256": EXPECTED["model_index"],
            "tokenizer_file_sha256": EXPECTED["tokenizer"],
            "training": "rank-32 LoRA only; base weights immutable",
            "target_layers": [20, 21, 22, 23],
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "trainable_elements": 4_528_128,
        },
        "datasets": datasets,
        "arms": _arms_v1(dataset_hashes),
        "compute_contract": {
            "mode": "compute_matched_quality",
            "target_gpu_seconds_per_arm": 14_400,
            "relative_match_tolerance": 0.02,
            "physical_gpu_ids": [0, 1, 2, 3],
            "useful_activity_each_gpu_required": True,
            "charged_gpu_second_definition": evaluation["compute_accounting"][
                "charged_gpu_second"
            ],
            "native_work_is_exact_and_gpu_seconds_are_acceptance_authority": True,
            "throughput_preflight_must_fit_every_fixed_stage_inside_its_target": True,
            "preflight_miss_action": "rebuild preregistration before model outputs; never adapt after outcomes",
        },
        "checkpoint_contract": {
            "initial_adapter_path": str(INITIAL_ADAPTER),
            "initial_adapter_state_sha256": EXPECTED["initial_state"],
            "initial_weights_file_sha256": EXPECTED["initial_weights"],
            "initial_config_file_sha256": EXPECTED["initial_config"],
            "initial_manifest_file_sha256": EXPECTED["initial_manifest"],
            "adapter_surface_sha256": EXPECTED["adapter_surface"],
            "cross_stage_transition": (
                "weights_only_exact_previous_output_new_optimizer_scheduler"
            ),
            "same_stage_resume_requires": [
                "exact_model_state",
                "exact_optimizer_state",
                "exact_scheduler_state",
                "exact_rng_state",
                "exact_dataloader_cursor",
            ],
            "checkpoint_receipt_requires": [
                "stage dataset content identity",
                "model state SHA-256",
                "optimizer/scheduler/RNG/dataloader state SHA-256",
                "completed native work and charged GPU seconds",
            ],
            "unregistered_checkpoint_fallback": "prohibited",
        },
        "evaluation_and_selection": {
            "registered_training_seeds": [1701, 1702, 1703],
            "knowledge": "dev equal-conflict-unit primary and generated QA tuple",
            "instruction_quality": "brief-answer exact/nonzero plus protocol-leak counters",
            "safety": "registered safety strata and non-degradation gates",
            "ood": evaluation["score_aggregation"]["ood_noninferiority"],
            "protected_holdout": (
                "one aggregate-only terminal access after curriculum recipe freeze"
            ),
            "protected_result_can_change_recipe": False,
        },
        "implementation_bindings": _implementation_bindings_v1(),
        "access_receipt": {
            "CPT_train_candidate_markdown_opened": True,
            "SFT_registered_train_semantics_opened": True,
            "dev_or_ood_semantics_opened": False,
            "protected_holdout_semantics_opened": False,
            "model_weights_loaded": False,
            "gpu_launched": False,
            "live_run_directory_touched": False,
        },
        "known_cpu_findings": {
            "registry_artifacts": 33,
            "rights_promoted_CPT_sources": 29,
            "rights_blocked_legacy_CPT_sources": [
                item["resource_id"] for item in cpt["excluded_registry_artifacts"]
            ],
            "rights_blocked_tokens": sum(
                item["available_tokens"] for item in cpt[
                    "excluded_registry_artifacts"
                ]
            ),
            "verified_SFT_rows": 433,
            "URL_trivia_rows_excluded": 15,
            "current_parent_contract_authorizes_curriculum": False,
        },
    }
    result["content_sha256_before_self_field"] = curriculum.canonical_sha256_v1(
        result
    )
    curriculum.validate_plan_v1(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)
    value = build_preregistration_v1()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v1(output),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
